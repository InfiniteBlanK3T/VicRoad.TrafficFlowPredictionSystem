import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from data.dataSCATS import process_data
from model import model
from keras.models import Model
warnings.filterwarnings("ignore")

def train_model(model, X_train, y_train, name, config):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """
    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05)

    model.save(f'model/trained/{name}.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(f'model/trained/{name}_loss.csv', encoding='utf-8', index=False)

def train_seas(models, X_train, y_train, name, config):
    """train
    train the SAEs model.

    # Arguments
        models: List, list of SAE model.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """
    temp = X_train

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(p.input,
                                       p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])

        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05)

        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer(f'hidden{i + 1}').set_weights(weights)

    train_model(saes, X_train, y_train, name, config)

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="lstm",
        help="Model to train: lstm, gru, bilstm, cnnlstm, saes")
    args = parser.parse_args()

    lag = 12
    config = {"batch": 256, "epochs": 3}
    file_path = 'data/Scats-Data-October-2006-Bundoora.csv'
    X_train, y_train, _, _, _ = process_data(file_path, lag)

    if args.model == 'lstm':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_lstm([12, 64, 64, 1])
    elif args.model == 'gru':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_gru([12, 64, 64, 1])
    elif args.model == 'bilstm':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_bidirectional_lstm([lag, 64, 64, 1])
    elif args.model == 'cnnlstm':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_cnn_lstm([lag, 64, 64, 1], n_steps=lag, n_features=1)
    elif args.model == 'saes':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
        m = model.get_saes([12, 400, 400, 400, 1])
        train_seas(m, X_train, y_train, args.model, config)
        return
    else:
        print(f"Unknown model: {args.model}")
        return

    train_model(m, X_train, y_train, args.model, config)

if __name__ == '__main__':
    main(sys.argv)