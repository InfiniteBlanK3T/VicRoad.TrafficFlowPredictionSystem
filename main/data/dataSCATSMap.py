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
import yaml

# Load configuration
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yml')

with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

""" WARNING PLEASE PROCEED WITH CARE IN CHANGING THIS FILE CODE """
""" ANY CHANGES IN THIS FILE MAY AFFECT THE WHOLE SYSTEM """


def process_data(file_path, lag=12, n_scats=None):
    """
    Process the SCATS data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing SCATS data.
        lag (int): Number of time steps to use for sequence data.
        n_scats (int, optional): Number of SCATS to process. If None, process all.
    
    Returns:
        tuple: Processed DataFrame, MinMaxScaler, and street segments dictionary.
    
    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
        ValueError: If the CSV file does not contain the expected columns.
    """
    try:
        # Load and preprocess the data
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The file {full_path} does not exist.")
        
        df = pd.read_csv(full_path, encoding='utf-8', header=0, skiprows=[0])
        
        if df.empty:
            raise pd.errors.EmptyDataError("The CSV file is empty.")
        
        required_columns = ['SCATS Number', 'Location', 'Date', 'NB_LATITUDE', 'NB_LONGITUDE']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("The CSV file does not contain all required columns.")
        
        # Convert date to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        
        # Melt the DataFrame to long format
        id_vars = ['SCATS Number', 'Location', 'Date', 'NB_LATITUDE', 'NB_LONGITUDE']
        value_vars = [f'V{str(i).zfill(2)}' for i in range(96)]
        df_melted = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='TimeSlot', value_name='TrafficVolume')
        
        # Create DateTime column
        df_melted['Time'] = pd.to_timedelta(df_melted['TimeSlot'].str[1:].astype(int) * 15, unit='m')
        df_melted['DateTime'] = df_melted['Date'] + df_melted['Time']
        df_melted = df_melted.sort_values(['SCATS Number', 'DateTime'])
        
        # Normalize traffic volume
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_melted['NormalizedVolume'] = scaler.fit_transform(df_melted['TrafficVolume'].values.reshape(-1, 1))
        
        # Extract street name
        df_melted['Street'] = df_melted['Location'].apply(lambda x: x.split('_')[0])
        
        # Create street segments
        street_segments = create_street_segments(df_melted)
        
        # Limit number of SCATS if specified
        if n_scats is not None:
            if n_scats == 'all':
                n_scats = None
            else:
                n_scats = int(n_scats)
                unique_scats = df_melted['SCATS Number'].unique()[:n_scats]
                df_melted = df_melted[df_melted['SCATS Number'].isin(unique_scats)]
        
        return df_melted, scaler, street_segments

    except Exception as e:
        logger.error(f"An error occurred while processing the data: {str(e)}")
        logger.error(f"File path: {file_path}")
        logger.error("Please check if the file exists and is accessible.")
        return None, None, None

def create_street_segments(df):
    """
    Create street segments from the processed DataFrame.
    
    Args:
        df (pd.DataFrame): Processed DataFrame containing SCATS data.
    
    Returns:
        dict: Street segments with coordinates and average traffic.
    
    Raises:
        ValueError: If the DataFrame is empty or does not contain required columns.
    """
    if df.empty:
        raise ValueError("The input DataFrame is empty.")
    
    required_columns = ['Street', 'NB_LATITUDE', 'NB_LONGITUDE', 'TrafficVolume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("The DataFrame does not contain all required columns.")
    
    street_segments = {}
    
    try:
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
    except Exception as e:
        logger.error(f"An error occurred while creating street segments: {str(e)}")
        return {}

def create_traffic_map(df, street_segments):
    """
    Create a folium map with traffic information.
    
    Args:
        df (pd.DataFrame): Processed DataFrame containing SCATS data.
        street_segments (dict): Street segments with coordinates and average traffic.
    
    Returns:
        folium.Map: A folium map object with traffic information.
    
    Raises:
        ValueError: If the input DataFrame or street_segments dictionary is empty.
    """
    if df.empty:
        raise ValueError("The input DataFrame is empty.")
    if not street_segments:
        raise ValueError("The street_segments dictionary is empty.")
    
    try:
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
    except Exception as e:
        logger.error(f"An error occurred while creating the traffic map: {str(e)}")
        return None

def get_color(normalized_value):
    """
    Get color based on normalized traffic value.
    
    Args:
        normalized_value (float): Normalized traffic value between 0 and 1.
    
    Returns:
        str: Color string for the traffic level.
    """
    if normalized_value < 0.25:
        return 'green'
    elif normalized_value < 0.5:
        return 'yellow'
    elif normalized_value < 0.75:
        return 'orange'
    else:
        return 'red'

def create_sequences(df, scats_number, sequence_length):
    """
    Create sequences for a specific SCATS number.
    
    Args:
        df (pd.DataFrame): Processed DataFrame containing SCATS data.
        scats_number (int): SCATS number to create sequences for.
        sequence_length (int): Length of input sequences.

    Returns:
        tuple: Input sequences and target values.
    
    Raises:
        ValueError: If the DataFrame is empty or does not contain the specified SCATS number.
    """
    if df.empty:
        raise ValueError("The input DataFrame is empty.")
    if scats_number not in df['SCATS Number'].unique():
        raise ValueError(f"SCATS number {scats_number} not found in the DataFrame.")
    
    try:
        data = df[df['SCATS Number'] == scats_number]['NormalizedVolume'].values
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)
    except Exception as e:
        logger.error(f"An error occurred while creating sequences: {str(e)}")
        return None, None

def prepare_model_data(df, sequence_length=12, test_size=0.2):
    """
    Prepare data for model training and testing.
    
    Args:
        df (pd.DataFrame): Processed DataFrame containing SCATS data.
        sequence_length (int): Length of input sequences.
        test_size (float): Proportion of data to use for testing.

    Returns:
        dict: Contains training and testing data for each SCATS number.
    
    Raises:
        ValueError: If the input DataFrame is empty.
    """
    if df.empty:
        raise ValueError("The input DataFrame is empty.")
    
    try:
        model_data = {}
        for scats_number in df['SCATS Number'].unique():
            X, y = create_sequences(df, scats_number, sequence_length)
            if X is not None and y is not None:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                model_data[scats_number] = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'y_test': y_test
                }
        return model_data
    except Exception as e:
        logger.error(f"An error occurred while preparing model data: {str(e)}")
        return {}