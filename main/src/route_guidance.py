import networkx as nx
import numpy as np
import math
import logging
import folium
from folium.plugins import MarkerCluster

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points using Haversine formula.
    
    Args:
        lat1, lon1: Coordinates of first point
        lat2, lon2: Coordinates of second point
        
    Returns:
        float: Distance in kilometers
    """
    R = 6371  # Earth's radius in kilometers

    # Convert latitude and longitude to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    distance = R * c  # Distance in kilometers
    return distance

def calculate_travel_time(volume: float, distance_km: float) -> float:
    """
    Calculate travel time based on volume and distance.
    
    Args:
        volume (float): Traffic volume (vehicles per 15 min)
        distance_km (float): Distance in kilometers
        
    Returns:
        float: Travel time in minutes
    """
    # Convert 15-min volume to hourly volume
    hourly_volume = volume * 4
    
    # Base speed parameters
    free_flow_speed = 60.0  # km/h (typical urban speed limit)
    min_speed = 20.0  # km/h (minimum speed in heavy congestion)
    capacity = 1800.0  # vehicles per hour per lane
    
    # Calculate congestion factor (0 to 1)
    congestion_factor = min(1.0, hourly_volume / capacity)
    
    # Calculate actual speed based on congestion
    # Speed decreases linearly with congestion
    actual_speed = max(min_speed, free_flow_speed * (1 - 0.67 * congestion_factor))
    
    # Calculate base travel time in minutes
    base_time = (distance_km / actual_speed) * 60
    
    # Add intersection delay (30 seconds per kilometer, as there are typically
    # more intersections on longer routes)
    intersection_delay = 0.5 * math.ceil(distance_km)
    
    # Add traffic light delay (average 1 minute per 2 kilometers)
    traffic_light_delay = 0.5 * math.ceil(distance_km)
    
    total_time = base_time + intersection_delay + traffic_light_delay
    
    return total_time

def create_graph(df, street_segments):
    """
    Create a graph representation of the street network.
    """
    if df.empty:
        raise ValueError("The input DataFrame is empty.")

    G = nx.Graph()

    try:
        # Add nodes (SCATS points)
        for scats, data in df.groupby("SCATS Number"):
            G.add_node(
                str(scats),
                pos=(data["NB_LATITUDE"].iloc[0], data["NB_LONGITUDE"].iloc[0])
            )

        # Connect nodes based on street segments
        for street, segment in street_segments.items():
            coords = segment["coords"]
            avg_traffic = segment["avg_traffic"]
            scats_on_street = df[df["Street"] == street]["SCATS Number"].unique()

            for i in range(len(scats_on_street) - 1):
                start = str(scats_on_street[i])
                end = str(scats_on_street[i + 1])
                
                if start in G and end in G:
                    # Get coordinates for start and end points
                    start_pos = G.nodes[start]["pos"]
                    end_pos = G.nodes[end]["pos"]
                    
                    # Calculate actual distance in kilometers
                    distance = calculate_distance_km(
                        start_pos[0], start_pos[1],
                        end_pos[0], end_pos[1]
                    )
                    
                    # Calculate travel time based on traffic volume and distance
                    travel_time = calculate_travel_time(avg_traffic, distance)
                    
                    # Add edge with both distance and time
                    G.add_edge(
                        start, end,
                        weight=travel_time,  # Weight is travel time for shortest path
                        distance=distance,
                        traffic=avg_traffic
                    )

        return G
    except Exception as e:
        logger.error(f"An error occurred while creating the graph: {str(e)}")
        return None

def find_routes(G, origin, destination, k=5):
    """
    Find k-shortest routes between origin and destination.
    """
    origin = str(origin)
    destination = str(destination)

    if origin not in G:
        raise ValueError(f"Origin SCATS number {origin} not found in the graph.")
    if destination not in G:
        raise ValueError(f"Destination SCATS number {destination} not found in the graph.")

    try:
        # Find k shortest paths
        routes = list(
            nx.shortest_simple_paths(G, origin, destination, weight="weight")
        )[:k]
        
        formatted_routes = []
        for route in routes:
            # Calculate total time and distance
            total_time = sum(
                G[route[i]][route[i + 1]]["weight"]
                for i in range(len(route) - 1)
            )
            total_distance = sum(
                G[route[i]][route[i + 1]]["distance"]
                for i in range(len(route) - 1)
            )
            
            # Add route information
            formatted_routes.append({
                "path": route,
                "time": total_time,
                "distance": total_distance
            })

        return formatted_routes
    except nx.NetworkXNoPath:
        raise ValueError(f"No path found between {origin} and {destination}.")

def route_guidance(df, street_segments, origin, destination):
    """
    Provide route guidance between origin and destination.
    """
    try:
        # Create the graph with accurate distances and travel times
        G = create_graph(df, street_segments)
        if G is None:
            raise ValueError("Failed to create the graph.")
        
        # Find routes
        routes = find_routes(G, origin, destination)
        
        # Sort routes by travel time
        routes.sort(key=lambda x: x['time'])
        
        return routes
    except Exception as e:
        logger.error(f"An error occurred in route guidance: {str(e)}")
        raise ValueError(f"Route guidance failed: {str(e)}")  
def create_route_map(df, route, origin, destination):
    """
    Create a folium map showing only the selected route from origin to destination.
    
    Args:
        df (pd.DataFrame): DataFrame containing SCATS data
        route (list): List of SCATS numbers in the route
        origin (str): Origin SCATS number
        destination (str): Destination SCATS number
        
    Returns:
        folium.Map: Map showing the route
    """
    try:
        # Get coordinates for the route
        route_coords = []
        for scats in route:
            scats_data = df[df["SCATS Number"] == int(scats)].iloc[0]
            route_coords.append([scats_data["NB_LATITUDE"], scats_data["NB_LONGITUDE"]])

        # Calculate center of the route
        center_lat = sum(coord[0] for coord in route_coords) / len(route_coords)
        center_lon = sum(coord[1] for coord in route_coords) / len(route_coords)

        # Create the map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

        # Add the route line
        folium.PolyLine(
            locations=route_coords,
            color="blue",
            weight=4,
            opacity=0.8
        ).add_to(m)

        # Add markers for origin and destination
        origin_data = df[df["SCATS Number"] == int(origin)].iloc[0]
        dest_data = df[df["SCATS Number"] == int(destination)].iloc[0]

        # Origin marker (green)
        folium.CircleMarker(
            location=[origin_data["NB_LATITUDE"], origin_data["NB_LONGITUDE"]],
            radius=8,
            popup=f"Origin: SCATS {origin}",
            color="green",
            fill=True
        ).add_to(m)

        # Destination marker (red)
        folium.CircleMarker(
            location=[dest_data["NB_LATITUDE"], dest_data["NB_LONGITUDE"]],
            radius=8,
            popup=f"Destination: SCATS {destination}",
            color="red",
            fill=True
        ).add_to(m)

        return m
    except Exception as e:
        logger.error(f"An error occurred while creating the route map: {str(e)}")
        return None
    
