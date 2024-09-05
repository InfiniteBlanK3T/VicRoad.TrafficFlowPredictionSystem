"""
Processing the data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

## Default function
# def process_data(train, test, lags):
#     """Process data
#     Reshape and split train\test data.

#     # Arguments
#         train: String, name of .csv train file.
#         test: String, name of .csv test file.
#         lags: integer, time lag.
#     # Returns
#         X_train: ndarray.
#         y_train: ndarray.
#         X_test: ndarray.
#         y_test: ndarray.
#         scaler: StandardScaler.
#     """
#     attr = 'Lane 1 Flow (Veh/5 Minutes)'
#     df1 = pd.read_csv(train, encoding='utf-8').fillna(0)
#     df2 = pd.read_csv(test, encoding='utf-8').fillna(0)

#     # scaler = StandardScaler().fit(df1[attr].values)
#     scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1[attr].values.reshape(-1, 1))
#     flow1 = scaler.transform(df1[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
#     flow2 = scaler.transform(df2[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

#     train, test = [], []
#     for i in range(lags, len(flow1)):
#         train.append(flow1[i - lags: i + 1])
#     for i in range(lags, len(flow2)):
#         test.append(flow2[i - lags: i + 1])

#     train = np.array(train)
#     test = np.array(test)
#     np.random.shuffle(train)

#     X_train = train[:, :-1]
#     y_train = train[:, -1]
#     X_test = test[:, :-1]
#     y_test = test[:, -1]

#     return X_train, y_train, X_test, y_test, scaler

## Making changes in order to read the file
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

# Usage example:
# X, y, scaler = process_data('path_to_your_csv_file.csv', lags=12)
# X_train, y_train, X_test, y_test = split_data(X, y)