# dataSCATSMap.py
## Overview
`dataSCATSMap.py` contains functions for processing SCATS data, creating street segments, and preparing data for model training.
## Key Functions

### 1. `process_data(file_path, lag=12, n_scats=None):`

- Processes SCATS data from a CSV file.
- Returns processed DataFrame, MinMaxScaler, and street segments dictionary.

### 2. `create_street_segments(df):`

- Creates street segments from the processed DataFrame.
- Returns a dictionary of street segments with coordinates and average traffic.

### 3. `create_traffic_map(df, street_segments):`

- Creates a folium map with traffic information.
- Returns a folium.Map object.

### 4. `create_sequences(df, scats_number, sequence_length):`

- Creates sequences for a specific SCATS number.
- Returns input sequences and target values.

### 5. `prepare_model_data(df, sequence_length=12, test_size=0.2):`

- Prepares data for model training and testing.
- Returns a dictionary containing training and testing data for each SCATS number.