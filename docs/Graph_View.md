# graph_view.py
## Overview

`graph_view.py` is the main GUI module of the Traffic Flow Prediction System. It creates and manages the graphical user interface for data visualization, prediction, and route guidance.

## Key Components

### 1. TFPSGUI Class: 

The main class that handles the GUI and system operations.

### 2. Initialization:

Sets up the main window and initializes data structures.
Calls `create_widgets()` and `show_loading_screen()`.

### 3. Data Loading:

- `show_loading_screen()`: Displays a loading screen with a progress bar.
- `load_data()`: Loads and processes SCATS data using functions from dataSCATSMap.py.
- `update_progress_bar()`: Updates the progress bar during data loading.
- `finish_loading()`: Finalizes the loading process and displays the main GUI.

### GUI Components:

- `create_widgets()`: Sets up the main notebook interface.
- `create_prediction_tab()`: Creates the traffic prediction tab.
- `create_route_guidance_map_tab()`: Creates the route guidance and map tab.

### Prediction Functionality:

- `predict()`: Performs traffic prediction based on user inputs.
- `prepare_input_data()`: Prepares input data for the prediction model.
- `plot_prediction()`: Plots the prediction results.

### Route Guidance and Map Visualization:

- `find_routes()`: Finds routes between selected origin and destination.
- `show_map()`: Displays the traffic map.
- `find_and_display_routes()`: Finds routes and displays them on the map.
- `update_map_with_route()`: Updates the map with a selected route.