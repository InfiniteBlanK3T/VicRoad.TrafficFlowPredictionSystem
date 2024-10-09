import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def process_data(file_path):
    """Process data
    Read and preprocess the SCATS data.

    # Arguments
        file_path: String, name of .csv file.
    # Returns
        df: pandas DataFrame with processed data.
        scaler: MinMaxScaler.
    """
    try:
        # Read the file as we intended
        df = pd.read_csv(file_path, encoding='utf-8', header=0, skiprows=[0])
        
        print("\nData read from csv file:")
        print(df.head(10))
        
        # Convert 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        
        # Melt the dataframe to long format
        id_vars = ['SCATS Number', 'Location', 'Date', 'NB_LATITUDE', 'NB_LONGITUDE']
        value_vars = [f'V{str(i).zfill(2)}' for i in range(96)]
        df_melted = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='TimeSlot', value_name='TrafficVolume')
        
        # Create a Time column
        df_melted['Time'] = pd.to_timedelta(df_melted['TimeSlot'].str[1:].astype(int) * 15, unit='m')
        
        # Create a datetime column
        df_melted['DateTime'] = df_melted['Date'] + df_melted['Time']
        
        # Sort the dataframe
        df_melted = df_melted.sort_values(['SCATS Number', 'DateTime'])
        
        # Normalize TrafficVolume
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_melted['NormalizedVolume'] = scaler.fit_transform(df_melted['TrafficVolume'].values.reshape(-1, 1))
        
        return df_melted, scaler

    except Exception as e:
        print(f"An error occurred while processing the data: {str(e)}")
        print(f"File path: {file_path}")
        print("Please check if the file exists and is accessible.")
        return None, None


def create_sequences(df, scats_number, sequence_length):
    """Create sequences for a specific SCATS number
    
    # Arguments
        df: pandas DataFrame with processed data.
        scats_number: int, SCATS number to create sequences for.
        sequence_length: int, length of input sequences.
    # Returns
        X: ndarray, input sequences.
        y: ndarray, target values.
    """
    data = df[df['SCATS Number'] == scats_number]['NormalizedVolume'].values
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def prepare_model_data(df, sequence_length=12, test_size=0.2):
    """Prepare data for model training and testing
    
    # Arguments
        df: pandas DataFrame with processed data.
        sequence_length: int, length of input sequences.
        test_size: float, proportion of data to use for testing.
    # Returns
        model_data: dict, containing training and testing data for each SCATS number.
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