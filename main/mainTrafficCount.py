"""
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).
"""
import math
import warnings
import numpy as np
import pandas as pd
from data.dataTrafficCount import process_data, split_data
from keras.models import load_model
from keras.utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
warnings.filterwarnings("ignore")


def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, ture data.
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
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
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
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%Y%m%d-%H%M")
    
    d = '2016-3-4 00:00'
    x = pd.date_range(d, periods=288, freq='5min')

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
    plt.savefig(f"images/TrafficCountLocation/{formatted_date_time}-graphTrafficCount.png")

def main():
    models = ['lstm', 'gru', 'bilstm', 'cnnlstm', 'saes']
    loaded_models = []
    for m in models:
        try:
            loaded_models.append(load_model(f'model/TrafficCountLocation/{m}.h5'))
        except:
            print(f"Could not load model: {m}")
    lag = 12

    
    ## Making changes in order to read the file
    file_path = 'data/Traffic_Count_Locations_with_LONG_LAT.csv'  
    X, y, scaler = process_data(file_path, lag)
    _, _, X_test, y_test = split_data(X, y)
    
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

    y_preds = []
    for name, model in zip(models, loaded_models):
        if name == 'saes':
            X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        elif name in ['lstm', 'gru', 'bilstm']:
            X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        elif name == 'cnnlstm':
            X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted = model.predict(X_test_reshaped)
        predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
        y_preds.append(predicted[:288])
        print(name)
        eva_regress(y_test, predicted)

    plot_results(y_test[: 288], y_preds, models)


if __name__ == '__main__':
    main()
