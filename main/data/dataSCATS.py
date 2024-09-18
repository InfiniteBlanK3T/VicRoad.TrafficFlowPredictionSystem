import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def process_data(file_path, lags):
    """Process data
    Reshape and split data.

    # Arguments
        file_path: String, name of .csv file.
        lags: integer, time lag.
    # Returns
        X_train: ndarray.
        y_train: ndarray.
        X_test: ndarray.
        y_test: ndarray.
        scaler: MinMaxScaler.
    """
    # Read the CSV file, skipping the first row (Start Time)
    df = pd.read_csv(file_path, encoding='utf-8', skiprows=[0]).fillna(0)
    
    # Extract flow data (columns V00 to V95)
    flow_columns = [f'V{str(i).zfill(2)}' for i in range(96)]
    flow_data = df[flow_columns].values.flatten()

    scaler = MinMaxScaler(feature_range=(0, 1))
    flow_normalized = scaler.fit_transform(flow_data.reshape(-1, 1)).reshape(1, -1)[0]

    sequences = []
    for i in range(lags, len(flow_normalized)):
        sequences.append(flow_normalized[i - lags: i + 1])

    data = np.array(sequences)
    np.random.shuffle(data)

    X = data[:, :-1]
    y = data[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test, scaler