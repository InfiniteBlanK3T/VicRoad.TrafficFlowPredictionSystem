import folium
from folium import plugins, Element
import logging
import pandas as pd
import osmnx as ox
from shapely.geometry import Point, LineString
import networkx as nx
import numpy as np
from branca.element import Figure
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

def calculate_map_center(df: pd.DataFrame, routes: List[Dict]) -> Tuple[float, float]:
    """Calculate the center point for the map based on all routes."""
    all_scats = set()
    for route in routes:
        all_scats.update(route['path'])
    
    route_points = df[df['SCATS Number'].isin([int(x) for x in all_scats])]
    return route_points['NB_LATITUDE'].mean(), route_points['NB_LONGITUDE'].mean()

def fit_map_bounds(m: folium.Map, df: pd.DataFrame, routes: List[Dict]):
    """Fit the map bounds to show all routes."""
    all_scats = set()
    for route in routes:
        all_scats.update(route['path'])
    
    route_points = df[df['SCATS Number'].isin([int(x) for x in all_scats])]
    bound_points = [
        [route_points['NB_LATITUDE'].min(), route_points['NB_LONGITUDE'].min()],
        [route_points['NB_LATITUDE'].max(), route_points['NB_LONGITUDE'].max()]
    ]
    m.fit_bounds(bound_points, padding=[50, 50])

def add_endpoints_to_map(m: folium.Map, df: pd.DataFrame, origin: str, destination: str):
    """Add origin and destination markers to the map."""
    origin_info = get_intersection_info(df, origin)
    dest_info = get_intersection_info(df, destination)

    # Origin marker
    folium.Marker(
        location=origin_info['coords'],
        popup=folium.Popup(
            f"<b>Start:</b> {origin_info['location']}<br>"
            f"<b>SCATS:</b> {origin_info['scats']}",
            max_width=300
        ),
        icon=folium.Icon(color='green', icon='info-sign')
    ).add_to(m)

    # Destination marker
    folium.Marker(
        location=dest_info['coords'],
        popup=folium.Popup(
            f"<b>End:</b> {dest_info['location']}<br>"
            f"<b>SCATS:</b> {dest_info['scats']}",
            max_width=300
        ),
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)

def format_intersection_name(location: str) -> str:
    """Format the intersection name from SCATS location."""
    try:
        roads = location.split('/')
        formatted_roads = []
        for road in roads:
            road = road.replace('_', ' ').title()
            road = road.replace(' Rd', ' Road').replace(' St', ' Street')
            formatted_roads.append(road)
        
        return ' & '.join(formatted_roads)
    except Exception as e:
        logger.error(f"Error formatting intersection name: {e}")
        return location

def get_intersection_info(df: pd.DataFrame, scats_number: str) -> Dict:
    """Get detailed intersection information for a SCATS number."""
    try:
        scats_data = df[df['SCATS Number'] == int(scats_number)].iloc[0]
        location = scats_data['Location']
        street_name = scats_data['Street']
        formatted_location = format_intersection_name(location)
        
        return {
            'scats': scats_number,
            'coords': [scats_data['NB_LATITUDE'], scats_data['NB_LONGITUDE']],
            'street': street_name,
            'location': formatted_location,
            'raw_location': location
        }
    except Exception as e:
        logger.error(f"Error getting intersection info for SCATS {scats_number}: {e}")
        return None

def get_route_geometry(df: pd.DataFrame, route: List[str], G: nx.Graph = None) -> List[List[float]]:
    """Get the actual street geometry for the route using OpenStreetMap data."""
    try:
        if G is None:
            route_points = df[df['SCATS Number'].isin([int(x) for x in route])]
            north = route_points['NB_LATITUDE'].max() + 0.01
            south = route_points['NB_LATITUDE'].min() - 0.01
            east = route_points['NB_LONGITUDE'].max() + 0.01
            west = route_points['NB_LONGITUDE'].min() - 0.01
            G = ox.graph_from_bbox(north, south, east, west, network_type='drive')
        
        route_coords = []
        for i in range(len(route) - 1):
            start_data = df[df['SCATS Number'] == int(route[i])].iloc[0]
            end_data = df[df['SCATS Number'] == int(route[i + 1])].iloc[0]
            
            start_node = ox.distance.nearest_nodes(
                G, 
                start_data['NB_LONGITUDE'], 
                start_data['NB_LATITUDE']
            )
            end_node = ox.distance.nearest_nodes(
                G,
                end_data['NB_LONGITUDE'],
                end_data['NB_LATITUDE']
            )
            
            try:
                path = nx.shortest_path(G, start_node, end_node, weight='length')
                path_coords = [[G.nodes[node]['y'], G.nodes[node]['x']] for node in path]
                route_coords.extend(path_coords)
            except nx.NetworkXNoPath:
                route_coords.extend([
                    [start_data['NB_LATITUDE'], start_data['NB_LONGITUDE']],
                    [end_data['NB_LATITUDE'], end_data['NB_LONGITUDE']]
                ])
        
        return route_coords
    except Exception as e:
        logger.error(f"Error getting route geometry: {e}")
        return [[df[df['SCATS Number'] == int(scats)].iloc[0]['NB_LATITUDE'],
                df[df['SCATS Number'] == int(scats)].iloc[0]['NB_LONGITUDE']]
               for scats in route]

def create_route_popup(route: Dict, df: pd.DataFrame) -> folium.Popup:
    """Create an interactive popup for a route."""
    route_points = []
    for scats in route['path']:
        info = get_intersection_info(df, scats)
        if info:
            route_points.append(info)

    popup_content = f"""
    <div style="min-width: 200px; max-width: 300px;">
        <h4 style="margin: 0 0 10px 0;">Route Details</h4>
        <div style="margin-bottom: 10px;">
            <strong>Distance:</strong> {route['distance']:.2f}km<br>
            <strong>Est. Time:</strong> {route['time']:.1f}min
        </div>
        <div>
            <strong>Route:</strong>
            <div style="margin-top: 5px;">
                <small>
                    {route_points[0]['location']} →<br>
                    {'<br>→ '.join(point['location'] for point in route_points[1:-1])}<br>
                    → {route_points[-1]['location']}
                </small>
            </div>
        </div>
    </div>
    """
    return folium.Popup(popup_content, max_width=300)

def add_route_to_map(m: folium.Map, df: pd.DataFrame, route: Dict, route_id: int, 
                    is_optimal: bool, G: nx.Graph = None):
    """Add a single route to the map with interactive features."""
    route_coords = get_route_geometry(df, route['path'], G)
    
    weight = 5 if is_optimal else 3
    opacity = 1.0 if is_optimal else 0.5
    z_index = 1000 if is_optimal else 100

    # Create a unique class for this route
    route_class = f"route-{route_id}"

    route_line = folium.PolyLine(
        locations=route_coords,
        color='blue' if is_optimal else 'gray',
        weight=weight,
        opacity=opacity,
        popup=create_route_popup(route, df),
        tooltip=f"Route {route_id + 1}: {route['distance']:.1f}km, {route['time']:.1f}min",
    )

    # Add the route line to the map
    route_line.add_to(m)

    # Add JavaScript to initialize this specific route
    m.get_root().html.add_child(Element(
        f"""
        <script>
        (function() {{
            // Wait for the map and elements to be ready
            setTimeout(function() {{
                try {{
                    var route = document.querySelector('path.leaflet-interactive:last-child');
                    if (route) {{
                        route.id = '{route_class}';
                        route.classList.add('route-line');
                        route.setAttribute('data-id', '{route_id}');
                        route.setAttribute('data-optimal', '{str(is_optimal).lower()}');
                        route.setAttribute('data-default-weight', '{weight}');
                        route.setAttribute('data-default-opacity', '{opacity}');
                        
                        // Add click handler
                        route.addEventListener('click', function() {{
                            highlightRoute('{route_id}');
                        }});
                        
                        // Add hover effects
                        route.addEventListener('mouseover', function() {{
                            if (!this.classList.contains('active')) {{
                                this.style.opacity = '0.8';
                                this.style.weight = '5';
                            }}
                        }});
                        
                        route.addEventListener('mouseout', function() {{
                            if (!this.classList.contains('active')) {{
                                this.style.opacity = '{opacity}';
                                this.style.weight = '{weight}';
                            }}
                        }});
                    }}
                }} catch (error) {{
                    console.error('Error initializing route:', error);
                }}
            }}, 100);
        }})();
        </script>
        """
    ))

    # Add waypoint markers for optimal route
    if is_optimal:
        for scats in route['path'][1:-1]:
            info = get_intersection_info(df, scats)
            if info:
                folium.CircleMarker(
                    location=info['coords'],
                    radius=6,
                    color='blue',
                    fill=True,
                    popup=folium.Popup(
                        f"<b>Via:</b> {info['location']}<br>"
                        f"<b>SCATS:</b> {info['scats']}",
                        max_width=300
                    )
                ).add_to(m)

def create_route_interaction_js() -> str:
    """Create JavaScript for route interaction."""
    return """
    function highlightRoute(routeId) {
        try {
            // Find all route elements
            var routes = document.querySelectorAll('.route-line');
            
            routes.forEach(function(route) {
                if (route.getAttribute('data-id') === routeId) {
                    // Highlight selected route
                    route.classList.add('active');
                    route.style.opacity = '1';
                    route.style.weight = '6';
                    route.style.zIndex = '1000';
                } else {
                    // Dim other routes
                    route.classList.remove('active');
                    route.style.opacity = '0.5';
                    route.style.weight = '3';
                    route.style.zIndex = '100';
                }
            });

            // Update summary panel
            var summaries = document.querySelectorAll('.route-summary');
            summaries.forEach(function(summary) {
                if (summary.getAttribute('data-route-id') === routeId) {
                    summary.classList.add('active');
                } else {
                    summary.classList.remove('active');
                }
            });
        } catch (error) {
            console.error('Error in highlightRoute:', error);
        }
    }

    function resetRoutes() {
        try {
            var routes = document.querySelectorAll('.route-line');
            routes.forEach(function(route) {
                route.classList.remove('active');
                route.style.opacity = route.getAttribute('data-default-opacity');
                route.style.weight = route.getAttribute('data-default-weight');
                
                // Highlight optimal route
                if (route.getAttribute('data-optimal') === 'true') {
                    route.classList.add('active');
                    route.style.opacity = '1';
                    route.style.weight = '5';
                }
            });

            // Reset summary panel
            var summaries = document.querySelectorAll('.route-summary');
            summaries.forEach(function(summary) {
                if (summary.getAttribute('data-optimal') === 'true') {
                    summary.classList.add('active');
                } else {
                    summary.classList.remove('active');
                }
            });
        } catch (error) {
            console.error('Error in resetRoutes:', error);
        }
    }

    // Initialize routes after map loads
    setTimeout(function() {
        try {
            // Highlight the optimal route by default
            var routes = document.querySelectorAll('.route-line');
            routes.forEach(function(route) {
                if (route.getAttribute('data-optimal') === 'true') {
                    route.classList.add('active');
                }
            });
        } catch (error) {
            console.error('Error in route initialization:', error);
        }
    }, 500);
    """

def create_multi_route_map(df: pd.DataFrame, routes: List[Dict], origin: str, destination: str) -> folium.Map:
    """Create an interactive map showing multiple routes."""
    try:
        center_lat, center_lon = calculate_map_center(df, routes)
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles='cartodbpositron'
        )

        # Add interactive styles and JavaScript
        m.get_root().html.add_child(Element(
            f"""
            <style>
                .route-summary {{
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 5px;
                    cursor: pointer;
                    transition: all 0.3s ease;
                }}
                .route-summary:hover {{
                    background-color: #f0f0f0;
                }}
                .route-summary.active {{
                    background-color: #e3f2fd;
                    border-left: 4px solid #1976d2;
                }}
                .route-details {{
                    font-size: 0.9em;
                    margin-top: 5px;
                }}
            </style>
            <script>
                {create_route_interaction_js()}
            </script>
            """
        ))

        # Get the bounding box for all routes
        all_scats = set()
        for route in routes:
            all_scats.update(route['path'])
        
        route_points = df[df['SCATS Number'].isin([int(x) for x in all_scats])]
        north = route_points['NB_LATITUDE'].max() + 0.01
        south = route_points['NB_LATITUDE'].min() - 0.01
        east = route_points['NB_LONGITUDE'].max() + 0.01
        west = route_points['NB_LONGITUDE'].min() - 0.01
        G = ox.graph_from_bbox(north, south, east, west, network_type='drive')

        # Add routes in reverse order (optimal route on top)
        for i, route in enumerate(routes):
            is_optimal = i == 0
            add_route_to_map(m, df, route, i, is_optimal, G)

        # Add markers and controls
        add_endpoints_to_map(m, df, origin, destination)
        add_route_summary_panel(m, routes, df)
        folium.LayerControl().add_to(m)
        plugins.Fullscreen().add_to(m)
        plugins.LocateControl().add_to(m)

        # Fit bounds
        fit_map_bounds(m, df, routes)

        return m

    except Exception as e:
        logger.error(f"Error creating interactive route map: {e}")
        return None

def add_route_summary_panel(m: folium.Map, routes: List[Dict], df: pd.DataFrame):
    """Add an interactive route summary panel to the map."""
    summary_html = """
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        z-index: 1000;
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        max-width: 300px;
        max-height: 400px;
        overflow-y: auto;
    ">
        <h4 style="margin-top: 0;">Available Routes</h4>
    """

    for i, route in enumerate(routes):
        is_optimal = i == 0
        via_points = [get_intersection_info(df, scats)['street'] 
                     for scats in route['path'][1:-1][:3]]
        
        summary_html += f"""
        <div id="summary-{i}" 
             class="route-summary{' active' if is_optimal else ''}"
             data-is-optimal="{str(is_optimal).lower()}"
             onclick="highlightRoute({i})"
             onmouseover="document.querySelector('#route-{i}').style.opacity = 0.8;"
             onmouseout="document.querySelector('#route-{i}').style.opacity = document.querySelector('#route-{i}').getAttribute('data-default-opacity');">
            <div style="color: {'#1976d2' if is_optimal else '#666'};">
                <strong>{f'🌟 Recommended Route' if is_optimal else f'Route {i+1}'}</strong>
            </div>
            <div class="route-details">
                <div>{route['distance']:.2f}km • {route['time']:.1f}min</div>
                <small style="color: #666;">
                    via {', '.join(via_points)}
                    {' ...' if len(route['path']) > 5 else ''}
                </small>
            </div>
        </div>
        <hr style="margin: 5px 0; border: none; border-top: 1px solid #eee;">
        """

    summary_html += """
    <div style="margin-top: 10px; text-align: center;">
        <small>
            <a href="#" onclick="resetRoutes(); return false;" 
               style="color: #666; text-decoration: none; hover: {color: #1976d2}">
               Reset View
            </a>
        </small>
    </div>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(summary_html))

def get_route_details(df: pd.DataFrame, route: Dict) -> str:
    """Get formatted details for a route."""
    points = []
    for scats in route['path']:
        info = get_intersection_info(df, scats)
        if info:
            points.append(info['location'])
    
    return f"""
    <div class="route-details-full">
        <p><strong>Distance:</strong> {route['distance']:.2f}km</p>
        <p><strong>Estimated Time:</strong> {route['time']:.1f}min</p>
        <p><strong>Route:</strong></p>
        <ol style="margin: 5px 0; padding-left: 20px;">
            <li>{points[0]} (Start)</li>
            {''.join(f'<li>{point}</li>' for point in points[1:-1])}
            <li>{points[-1]} (End)</li>
        </ol>
    </div>
    """

def create_interactive_elements() -> str:
    """Create CSS and JavaScript for interactive elements."""
    return """
    <style>
        .route-line {
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .route-line:hover {
            stroke-width: 6px;
        }
        .route-line.active {
            stroke-width: 6px;
            stroke-opacity: 1;
        }
        .route-summary {
            transition: all 0.3s ease;
        }
        .route-summary:hover {
            transform: translateX(5px);
        }
        .popup-content {
            font-family: Arial, sans-serif;
            line-height: 1.4;
        }
        .popup-content h4 {
            color: #1976d2;
            margin-bottom: 10px;
        }
    </style>
    """

def format_time(minutes: float) -> str:
    """Format time in minutes to a readable string."""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    if hours > 0:
        return f"{hours}h {mins}min"
    return f"{mins}min"

def calculate_bounds(df: pd.DataFrame, routes: List[Dict]) -> List[List[float]]:
    """Calculate map bounds that encompass all routes."""
    all_scats = set()
    for route in routes:
        all_scats.update(route['path'])
    
    points = df[df['SCATS Number'].isin([int(x) for x in all_scats])]
    padding = 0.01  # About 1km padding
    
    return [
        [points['NB_LATITUDE'].min() - padding, points['NB_LONGITUDE'].min() - padding],
        [points['NB_LATITUDE'].max() + padding, points['NB_LONGITUDE'].max() + padding]
    ]

def generate_route_colors(num_routes: int) -> List[str]:
    """Generate a list of visually distinct colors for routes."""
    if num_routes <= 1:
        return ['#1976d2']  # Blue for single/optimal route
    
    colors = ['#1976d2']  # Optimal route is always blue
    base_colors = ['#757575', '#9e9e9e', '#bdbdbd', '#e0e0e0']  # Gray scale for alternatives
    
    # Cycle through base colors for additional routes
    for i in range(num_routes - 1):
        colors.append(base_colors[i % len(base_colors)])
    
    return colors