import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import folium
from folium.plugins import MarkerCluster
import logging
import osmnx as ox
import networkx as nx
from shapely.geometry import LineString, Point
from geopandas import GeoDataFrame
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_data(file_path):
    try:
        # Use os.path.join for cross-platform compatibility
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        df = pd.read_csv(full_path, encoding='utf-8', header=0, skiprows=[0])
        
        logger.info("Data read from csv file:")
        logger.info(df.head(10))
        
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        
        # Melt the dataframe to long format
        id_vars = ['SCATS Number', 'Location', 'Date', 'NB_LATITUDE', 'NB_LONGITUDE']
        value_vars = [f'V{str(i).zfill(2)}' for i in range(96)]
        df_melted = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='TimeSlot', value_name='TrafficVolume')
        
        df_melted['Time'] = pd.to_timedelta(df_melted['TimeSlot'].str[1:].astype(int) * 15, unit='m')
        df_melted['DateTime'] = df_melted['Date'] + df_melted['Time']
        df_melted = df_melted.sort_values(['SCATS Number', 'DateTime'])
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_melted['NormalizedVolume'] = scaler.fit_transform(df_melted['TrafficVolume'].values.reshape(-1, 1))
        
        # Extract street names and create street segments
        df_melted['Street'] = df_melted['Location'].apply(lambda x: x.split('_')[0])
        street_segments = create_street_segments(df_melted)
        
        return df_melted, scaler, street_segments

    except Exception as e:
        logger.error(f"An error occurred while processing the data: {str(e)}")
        logger.error(f"File path: {file_path}")
        logger.error("Please check if the file exists and is accessible.")
        return None, None, None

def create_street_segments(df):
    street_segments = {}
    
    # Get the bounding box of all coordinates
    north, south = df['NB_LATITUDE'].max(), df['NB_LATITUDE'].min()
    east, west = df['NB_LONGITUDE'].max(), df['NB_LONGITUDE'].min()
    
    # Download the street network for the area
    G = ox.graph_from_bbox(north, south, east, west, network_type='drive')
    
    for street in df['Street'].unique():
        street_data = df[df['Street'] == street]
        coords = street_data[['NB_LATITUDE', 'NB_LONGITUDE']].drop_duplicates().values
        
        if len(coords) >= 2:
            avg_traffic = street_data['TrafficVolume'].mean()
            
            # Find the nearest network edges for the street
            path = []
            for i in range(len(coords) - 1):
                start_node = ox.distance.nearest_nodes(G, coords[i][1], coords[i][0])
                end_node = ox.distance.nearest_nodes(G, coords[i+1][1], coords[i+1][0])
                try:
                    path += nx.shortest_path(G, start_node, end_node, weight='length')
                except nx.NetworkXNoPath:
                    logger.warning(f"No path found for {street} between points {i} and {i+1}")
            
            if path:
                # Get the coordinates of the path
                path_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in path]
                
                street_segments[street] = {
                    'coords': path_coords,
                    'avg_traffic': avg_traffic
                }
    return street_segments

def create_traffic_map(df, street_segments):
    center_lat = df['NB_LATITUDE'].mean()
    center_lon = df['NB_LONGITUDE'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    marker_cluster = MarkerCluster().add_to(m)
    
    max_traffic = max(segment['avg_traffic'] for segment in street_segments.values())
    min_traffic = min(segment['avg_traffic'] for segment in street_segments.values())
    
    # Add lines for street segments with color indicating traffic flow
    for street, data in street_segments.items():
        coords = data['coords']
        avg_traffic = data['avg_traffic']
        normalized_traffic = (avg_traffic - min_traffic) / (max_traffic - min_traffic)
        color = get_color(normalized_traffic)
        
        folium.PolyLine(
            locations=coords,
            color=color,
            weight=4,
            opacity=0.8,
            popup=f"Street: {street}<br>Avg Traffic: {avg_traffic:.2f}"
        ).add_to(m)
    
    # Add markers for intersections
    for _, row in df.groupby(['SCATS Number', 'Location', 'NB_LATITUDE', 'NB_LONGITUDE'])['TrafficVolume'].mean().reset_index().iterrows():
        folium.CircleMarker(
            location=[row['NB_LATITUDE'], row['NB_LONGITUDE']],
            radius=5,
            popup=f"SCATS: {row['SCATS Number']}<br>Location: {row['Location']}<br>Avg Traffic: {row['TrafficVolume']:.2f}",
            color='black',
            fill=True,
            fillColor='black'
        ).add_to(marker_cluster)
    
    return m

def get_color(normalized_value):
    if normalized_value < 0.25:
        return 'green'
    elif normalized_value < 0.5:
        return 'yellow'
    elif normalized_value < 0.75:
        return 'orange'
    else:
        return 'red'
    
## These 2 are very good and working so far dont change them unless you know what you are doing!!
def create_sequences(df, scats_number, sequence_length):
    """
    Create sequences for a specific SCATS number.
    
    Args:
        df (pd.DataFrame): Processed DataFrame.
        scats_number (int): SCATS number to create sequences for.
        sequence_length (int): Length of input sequences.

    Returns:
        tuple: Input sequences and target values.
    """
    data = df[df['SCATS Number'] == scats_number]['NormalizedVolume'].values
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def prepare_model_data(df, sequence_length=12, test_size=0.2):
    """
    Prepare data for model training and testing.
    
    Args:
        df (pd.DataFrame): Processed DataFrame.
        sequence_length (int): Length of input sequences.
        test_size (float): Proportion of data to use for testing.

    Returns:
        dict: Contains training and testing data for each SCATS number.
    """
    model_data = {}
    for scats_number in df['SCATS Number'].unique():
        X, y = create_sequences(df, scats_number, sequence_length)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        model_data[scats_number] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }
    return model_data
