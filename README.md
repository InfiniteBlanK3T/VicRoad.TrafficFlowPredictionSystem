# Traffic Flow Prediction System (TFPS)

This project implements a traffic flow prediction system using machine learning techniques 
COS30018-Intelligent Systems - Semester 2 2024

## Preliminary
### Prerequisite - Installation:
#### 1. Installing `pyenv` - Python virtual environment: [For more info of PyENV](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation)
 - `curl https://pyenv.run | bash`
#### Adding `pyenv` to your shell configuration (`~/.bashrc` or `~/.zshrc`)
> [!WARNING]  
> Your PC may need `build-essential` and these dependencies
- `sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git`

#### 2. Installing `PYENV` Version:
1. Install pyenv (python virtual environment) python 3.9 or in my case I use 3.10.12 it works fine:
- `pyenv install <python version here>`
- `pyenv install 3.10.12`

## Development Set Up
### Manual Process:
1. Create Virtual Environment Directory - with Python version installed above:
> [!IMPORTANT]  
> It is important that you activate your virtual environment first before installing dependencies `source .venv/bin/activate`

(.venv - what ever name you like I like .venv) in which you would put all your dependencies in this Directory here:
- `mkdir .venv`

2. Set the version you want to install - recommend 3.10.12 it works on my PC:
`pyenv local 3.10.12`
3. Installing Python Virtual Environment onto that `.venv` folder:
`pyenv exec python -m venv .venv`
4. Activate virtual environment
`source .venv/bin/activate`
5. Installing all the dependencies from requirements.txt
`pip install -r requirements.txt`
### Automation
I have created `scripts/setup_dev_env.sh` to automate process above assuming that you already followed [Prerequisite - Installation](#prerequisite---installation)
1. Adding permission to run the script if running for first time:
`chmod +x ./scripts/setup_dev_env.sh`
2. Running the script automate proces above:
`./scripts/setup_dev_env.sh`

## Graphs
### v0.2 12th Oct
![Screenshot 2024-10-17 at 14 39 55](https://github.com/user-attachments/assets/1eae5d94-8633-4079-aa73-705b5f64a7ab)

### v0.1 19th Sept
![image](https://github.com/user-attachments/assets/07dc703b-ee41-48f0-b28b-8e47fb54bfd0)


## Project Structure

- `data/`: Contains datasets
- `models/`: Stores trained models
- `src/`: Source code for the TFPS
- `tests/`: Unit tests
- `docs/`: Project documentation

## Usage
Project has 3 main options:

### 1. Training
> [!INFO]
> Please make sure to add permission to run it with `chmod +x main/scripts/train.sh`
The `train_model.py` or with script `main/scripts/train.sh` trains machine learning models and saves their weights and loss history.
#### Summary
- **Models**: LSTM, GRU, SAES, BiLSTM, CNN-LSTM
- **Training History**: Saves training history including loss and MAPE to a CSV file.
- **Model Weights**: Saves trained model weights with filenames that include the current timestamp.

For detailed information, refer to [Training Model Documentation](https://github.com/InfiniteBlanK3T/VicRoad.TrafficFlowPredictionSystem/blob/main/docs/TrainingModel.md)

#### Usage
To train the models, use the following command:
Syntax:
`python main/src/train_model.py --model <model-you-want> --n_scats <numbers here 1-141 or 'all'>`
Example:
- `python main/src/train_model.py --model gru --n_scats 5`

Ensure that your config.yml file is correctly set up with the appropriate paths for saving the trained models and loss history.
```bash
training:
  models: ['lstm', 'gru', 'bilstm', 'cnnlstm', 'saes']
  model_save_path: 'main/model/trained'
```

### 2. Main GUI interactive
The `graph_view.py` and `dataSCATSMap.py` scripts provide a graphical user interface (GUI) for the Traffic Flow Prediction System (TFPS).

#### Summary
Data Loading: Load and process SCATS data.
Model Setup: Load trained models for each SCATS number.
Traffic Prediction: Predict traffic volumes based on user inputs.
Route Guidance: Find and display routes between selected origin and destination SCATS numbers.
Map Visualization: Display traffic maps and routes.
#### Usage
To run the GUI, use the following command:
`python main/graph_view.py`
Ensure that your config.yml file is correctly set up with the appropriate paths and settings.
> [!INFO]
> For more accuracy in prediction recommend epochs around: `600` instead of `3`

```bash
data:
  file_path: 'Scats-Data-October-2006-Bundoora.csv'
  lag: 12
  n_scats_options: [5, 10, 'all']

gui:
  window_size: '1400x800'

model:
  batch_size: 256
  epochs: 3
  validation_split: 0.05
```

#### Interaction between Modules

1. Data Processing: `graph_view.py` uses `process_data()` from [dataSCATSMap.py] to load and process SCATS data.
2. Model Data Preparation: `prepare_model_data()` from [dataSCATSMap.py] is used in [graph_view.py] to prepare data for model training and prediction.
3. Map Creation: `create_traffic_map()` from [dataSCATSMap.py] is used in [graph_view.py] to create and display the traffic map.
4. Street Segments: The street segments created by `create_street_segments()` in [dataSCATSMap.py] are used in [graph_view.py] for route guidance and map visualization.

For detailed infromation:
- Refer to [graph_view.py Documentation](https://github.com/InfiniteBlanK3T/VicRoad.TrafficFlowPredictionSystem/blob/main/docs/Graph_View.md)
- Refer to [dataSCATSMap.py Documentation](https://github.com/InfiniteBlanK3T/VicRoad.TrafficFlowPredictionSystem/blob/main/docs/dataSCATSMap.md)
### 3. Report
The main_evaluation.py script generates various reports and visualizations to evaluate model performance.

#### Summary
Metric Comparison Plots: Generates .png files comparing metrics like EVS, MAPE, MSE, R2, and RMSE across models.
Model Performance Comparison: Comprehensive plot comparing all models across all metrics.
Prediction Plots: Plots showing predictions of all models for each SCATS number.
Comparison Report: Markdown report summarizing model performance, including overall performance, best model for each metric, and performance by SCATS.
#### Running code:
To run the evaluation and generate the reports, use the following command:
`python main/main_evaluation.py`
Ensure that your config.yml file is correctly set up with the appropriate paths for saving the reports and plots.
```bash
evaluation:
  metric_save_path: 'reports/metric-comparison/'
  prediction_save_path: 'reports/prediction-comparison/'
  report_summary_save_path: 'reports/summary/'
```
For detailed information, refer to [Report Documentation](https://github.com/InfiniteBlanK3T/VicRoad.TrafficFlowPredictionSystem/blob/main/docs/Report.md)

## Contributing



## License

