"""
Processing the data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def process_data(file_path, lags):
    """Process data
    Reshape and split train\test data.
    # Arguments
    file_path: String, path to the .csv file.
    lags: integer, time lag.
    # Returns
    X: ndarray, input features.
    y: ndarray, target values.
    scaler: MinMaxScaler.
    """
    # Read the CSV file
    df = pd.read_csv(file_path, encoding='utf-8')
    
    # Identify the traffic flow columns (assuming they are numeric columns)
    flow_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Extract the traffic flow data
    flow_data = df[flow_columns].values.flatten()
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    flow_normalized = scaler.fit_transform(flow_data.reshape(-1, 1)).flatten()
    
    # Create sequences
    sequences = []
    for i in range(lags, len(flow_normalized)):
        sequences.append(flow_normalized[i-lags:i+1])
    
    # Convert to numpy array and shuffle
    sequences = np.array(sequences)
    np.random.shuffle(sequences)
    
    # Split into features (X) and target (y)
    X = sequences[:, :-1]
    y = sequences[:, -1]
    
    return X, y, scaler

def split_data(X, y, test_split=0.2):
    """Split data into train and test sets"""
    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, y_train, X_test, y_test