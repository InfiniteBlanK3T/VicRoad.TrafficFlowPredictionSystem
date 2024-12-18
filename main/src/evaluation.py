import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from keras.models import load_model
from data.dataSCATSMap import process_data, prepare_model_data
import yaml
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# Load configuration
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yml')

with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)


# Create necessary directories
os.makedirs(config['training']['model_save_path'], exist_ok=True)
os.makedirs(config['evaluation']['prediction_save_path'], exist_ok=True)

def MAPE(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.
    
    Returns:
        float: MAPE value.
    """
    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]
    return np.mean(np.abs((np.array(y) - np.array(y_pred)) / np.array(y))) * 100

def evaluate_regression(y_true, y_pred):
    """
    Evaluate regression metrics.
    
    Args:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.
    
    Returns:
        dict: Dictionary of evaluation metrics.
    """
    try:
        mape = MAPE(y_true, y_pred)
        evs = explained_variance_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MAPE': mape,
            'EVS': evs,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
    except Exception as e:
        logger.error(f"Error in evaluate_regression: {str(e)}")
        raise

def plot_results(y_true, y_preds, names, scats_number):
    """
    Plot a summary of all model predictions.
    
    Args:
        y_true (array-like): True values.
        y_preds (list): List of predicted values for each model.
        names (list): List of model names.
        scats_number (int): SCATS number.
    """
    try:
        sns.set_style("whitegrid")
        plt.figure(figsize=(16, 10))

        start_date = datetime(2006, 10, 1)
        x = [start_date + timedelta(minutes=15*i) for i in range(96)]

        plt.plot(x, y_true, label='True Data', color='black', linewidth=2)

        colors = sns.color_palette("husl", len(names))
        for name, y_pred, color in zip(names, y_preds, colors):
            plt.plot(x, y_pred, label=name, color=color, linewidth=2, alpha=0.7)

        plt.legend(loc='upper left', fontsize=12)
        plt.title(f'Traffic Flow Prediction - All Models (SCATS {scats_number})', fontsize=20)
        plt.xlabel('Time of Day', fontsize=14)
        plt.ylabel('Traffic Flow', fontsize=14)

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=3))
        plt.gcf().autofmt_xdate()

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        plt.savefig(f"{config['evaluation']['prediction_save_path']}/{current_time}-ScatsData_all_predictions_{scats_number}.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Error in plot_results: {str(e)}")
        raise

def plot_individual_results(y_true, y_preds, names, scats_number):
    """
    Plot individual results for each model.
    
    Args:
        y_true (array-like): True values.
        y_preds (list): List of predicted values for each model.
        names (list): List of model names.
        scats_number (int): SCATS number.
    """
    try:
        sns.set_style("whitegrid")
        fig, axs = plt.subplots(len(names), 1, figsize=(16, 6*len(names)), sharex=True)
        
        start_date = datetime(2006, 10, 1)
        x = [start_date + timedelta(minutes=15*i) for i in range(96)]

        colors = sns.color_palette("husl", len(names))

        for i, (name, y_pred, color) in enumerate(zip(names, y_preds, colors)):
            axs[i].plot(x, y_true, label='True Data', color='black', linewidth=2, alpha=0.7)
            axs[i].plot(x, y_pred, label=name, color=color, linewidth=2)
            axs[i].set_title(f'{name} Prediction vs True Data (SCATS {scats_number})', fontsize=16)
            axs[i].set_ylabel('Traffic Flow', fontsize=12)
            axs[i].legend(loc='upper left', fontsize=10)
            axs[i].grid(True, linestyle='--', alpha=0.7)

            # Calculate and display MAPE on the plot
            mape = MAPE(y_true, y_pred)
            axs[i].text(0.02, 0.95, f'MAPE: {mape:.2f}%', transform=axs[i].transAxes, 
                        fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

        plt.xlabel('Time of Day', fontsize=14)
        fig.autofmt_xdate()
        
        for ax in axs:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))

        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        plt.savefig(f"{config['evaluation']['prediction_save_path']}/{current_time}-ScatsData_individual_predictions_{scats_number}.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Error in plot_individual_results: {str(e)}")
        raise

def evaluate_models(file_path, lag=12, n_scats=None):
    """
    Evaluate all models for given SCATS data.
    
    Args:
        file_path (str): Path to the SCATS data file.
        lag (int): Lag for time series data.
        n_scats (int): Number of SCATS to evaluate.
    
    Returns:
        dict: Evaluation results for all models and SCATS.
    """
    try:
        models = config['training']['models']
        loaded_models = {model_name: {} for model_name in models}
        
        df_melted, scaler, _ = process_data(file_path, lag, n_scats=n_scats)
        model_data = prepare_model_data(df_melted, sequence_length=lag)

        results = {}

        for scats_number, data in model_data.items():
            X_test, y_test = data['X_test'], data['y_test']
            y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

            y_preds = []
            results[scats_number] = {}

            for name in models:
                try:
                    model_path = f"{config['training']['model_save_path']}/{name}_{scats_number}.h5"
                    if scats_number not in loaded_models[name]:
                        loaded_models[name][scats_number] = load_model(model_path)
                    model = loaded_models[name][scats_number]
                except Exception as e:
                    logger.warning(f"Could not load model {name} for SCATS {scats_number}: {e}")
                    continue

                if name == 'saes':
                    X_test_reshaped = X_test
                elif name in ['lstm', 'gru', 'bilstm', 'cnnlstm']:
                    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                else:
                    logger.warning(f"Unknown model type: {name}")
                    continue

                predicted = model.predict(X_test_reshaped)
                predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
                y_preds.append(predicted[:96])
                
                metrics = evaluate_regression(y_test, predicted)
                results[scats_number][name] = metrics

            plot_results(y_test[:96], y_preds, models, scats_number)
            plot_individual_results(y_test[:96], y_preds, models, scats_number)

        return results
    except Exception as e:
        logger.error(f"Error in evaluate_models: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        file_path = config['data']['file_path']
        lag = config['data']['lag']
        n_scats = config['data']['n_scats_options'][0]  # Use the first option as default
        results = evaluate_models(file_path, lag, n_scats)
        
        # Save results to a JSON file
        with open('evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info("Evaluation completed successfully. Results saved to evaluation_results.json")
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}")