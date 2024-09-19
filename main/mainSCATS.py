import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from keras.models import load_model
from data.dataSCATS import process_data

warnings.filterwarnings("ignore")

def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]
    return np.mean(np.abs((np.array(y) - np.array(y_pred)) / np.array(y))) * 100

def evaluate_regression(y_true, y_pred):
    """Evaluate regression metrics"""
    mape = MAPE(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f'Explained variance score: {evs:.6f}')
    print(f'MAPE: {mape:.6f}%')
    print(f'MAE: {mae:.6f}')
    print(f'MSE: {mse:.6f}')
    print(f'RMSE: {rmse:.6f}')
    print(f'R2: {r2:.6f}')

def plot_results(y_true, y_preds, names):
    """Plot a summary of all model predictions"""
    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 10))

    start_date = datetime(2006, 10, 1)
    x = [start_date + timedelta(minutes=15*i) for i in range(96)]

    plt.plot(x, y_true, label='True Data', color='black', linewidth=2)

    colors = sns.color_palette("husl", len(names))
    for name, y_pred, color in zip(names, y_preds, colors):
        plt.plot(x, y_pred, label=name, color=color, linewidth=2, alpha=0.7)

    plt.legend(loc='upper left', fontsize=12)
    plt.title('Traffic Flow Prediction - All Models', fontsize=20)
    plt.xlabel('Time of Day', fontsize=14)
    plt.ylabel('Traffic Flow', fontsize=14)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=3))
    plt.gcf().autofmt_xdate()

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    plt.savefig(f"images/ScatsData-Bundoora/{current_time}-ScatsData_all_predictions.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_individual_results(y_true, y_preds, names):
    """Plot individual results for each model"""
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(len(names), 1, figsize=(16, 6*len(names)), sharex=True)
    
    start_date = datetime(2006, 10, 1)
    x = [start_date + timedelta(minutes=15*i) for i in range(96)]

    colors = sns.color_palette("husl", len(names))

    for i, (name, y_pred, color) in enumerate(zip(names, y_preds, colors)):
        axs[i].plot(x, y_true, label='True Data', color='black', linewidth=2, alpha=0.7)
        axs[i].plot(x, y_pred, label=name, color=color, linewidth=2)
        axs[i].set_title(f'{name} Prediction vs True Data', fontsize=16)
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

    plt.tight_layout()
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    plt.savefig(f"images/ScatsData-Bundoora/{current_time}-ScatsData_individual_predictions.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    models = ['lstm', 'gru', 'bilstm', 'cnnlstm', 'saes']
    loaded_models = []
    for m in models:
        try:
            loaded_models.append(load_model(f'model/ScatsData-Bundoora/{m}.h5'))
        except Exception as e:
            print(f"Could not load model {m}: {e}")

    lag = 12
    file_path = 'data/Scats-Data-October-2006-Bundoora.csv'
    _, _, X_test, y_test, scaler = process_data(file_path, lag)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

    y_preds = []
    for name, model in zip(models, loaded_models):
        if name == 'saes':
            X_test_reshaped = X_test
        elif name in ['lstm', 'gru', 'bilstm', 'cnnlstm']:
            X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        else:
            print(f"Unknown model type: {name}")
            continue

        predicted = model.predict(X_test_reshaped)
        predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
        y_preds.append(predicted[:96])
        print(name.upper())
        evaluate_regression(y_test, predicted)
        print()

    plot_results(y_test[:96], y_preds, models)
    plot_individual_results(y_test[:96], y_preds, models)

if __name__ == '__main__':
    main()