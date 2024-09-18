"""
Traffic Flow Prediction with Neural Networks (SAEs, LSTM, GRU).
"""
import math
import warnings
import numpy as np
import pandas as pd
from data.dataSCATS import process_data
from keras.models import load_model
from keras.utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
warnings.filterwarnings("ignore")

def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, true data.
        y_pred: List/ndarray, predicted data.
    # Returns
        mape: Double, result data for train.
    """
    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape

def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted result.

    # Arguments
        y_true: List/ndarray, true data.
        y_pred: List/ndarray, predicted data.
    """
    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)


def plot_results(y_true, y_preds, names):
    """Plot
    Plot the true data and predicted data with improved presentation.

    # Arguments
        y_true: List/ndarray, true data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 10))

    # Create x-axis dates
    start_date = datetime(2006, 10, 1)
    x = [start_date + timedelta(minutes=15*i) for i in range(96)]

    # Plot true data
    plt.plot(x, y_true, label='True Data', color='black', linewidth=2)

    # Color palette for predictions
    colors = sns.color_palette("husl", len(names))

    # Plot predictions
    for name, y_pred, color in zip(names, y_preds, colors):
        plt.plot(x, y_pred, label=name, color=color, linewidth=2, alpha=0.7)

    plt.legend(loc='upper left', fontsize=12)
    plt.title('Traffic Flow Prediction Comparison', fontsize=20)
    plt.xlabel('Time of Day', fontsize=14)
    plt.ylabel('Traffic Flow', fontsize=14)

    # Improve x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=3))
    plt.gcf().autofmt_xdate()

    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()

    plt.savefig("images/ScatsData-Bundoora/ScatsData_prediction_improved.png", dpi=300, bbox_inches='tight')
    
    plt.show()

    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, true data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
    d = '2006-10-1 00:00'
    x = pd.date_range(d, periods=96, freq='15min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()
    plt.savefig("images/ScatsData-Bundoora/ScatsData_prediction.png")

def plot_individual_results(y_true, y_preds, names):
    """Plot individual results for each model.

    # Arguments
        y_true: List/ndarray, true data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
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

    plt.xlabel('Time of Day', fontsize=14)
    fig.autofmt_xdate()
    
    for ax in axs:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))

    plt.tight_layout()
    plt.savefig("images/ScatsData-Bundoora/ScatsData_individual_predictions.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    lstm = load_model('model/ScatsData-Bundoora/lstm.h5')
    gru = load_model('model/ScatsData-Bundoora/gru.h5')
    saes = load_model('model/ScatsData-Bundoora/saes.h5')
    models = [lstm, gru, saes]
    names = ['LSTM', 'GRU', 'SAEs']

    lag = 12
    file_path = 'data/Scats-Data-October-2006-Bundoora.csv'
    _, _, X_test, y_test, scaler = process_data(file_path, lag)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

    y_preds = []
    for name, model in zip(names, models):
        if name == 'SAEs':
            X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        else:
            X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        file = f'images/{name}_model.png'
        plot_model(model, to_file=file, show_shapes=True)
        predicted = model.predict(X_test_reshaped)
        predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
        y_preds.append(predicted[:96])
        print(name)
        eva_regress(y_test, predicted)

    plot_results(y_test[:96], y_preds, names)
    
    plot_individual_results(y_test[:96], y_preds, names)

if __name__ == '__main__':
    main()