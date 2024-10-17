import sys
import os
from pathlib import Path
# Add the root directory to the Python path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import yaml
import argparse
import numpy as np
import pandas as pd
from data.dataSCATSMap import process_data, prepare_model_data
from model import model
from keras.models import Model

# Load configuration
with open('config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

def train_model(model, X_train, y_train, name, scats, config):
    """Train a single model."""
    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    hist = model.fit(
        X_train, y_train,
        batch_size=config['model']['batch_size'],
        epochs=config['model']['epochs'],
        validation_split=config['model']['validation_split'])

    model_save_path = f"{config['training']['model_save_path']}/{name}_{scats}.h5"
    model.save(model_save_path)
    pd.DataFrame.from_dict(hist.history).to_csv(f"{config['training']['model_save_path']}/{name}_{scats}_loss.csv", encoding='utf-8', index=False)

def train_seas(models, X_train, y_train, name, scats, config):
    """Train the SAEs model."""
    temp = X_train

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(p.input, p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
        m.fit(temp, y_train, batch_size=config['model']['batch_size'],
              epochs=config['model']['epochs'],
              validation_split=config['model']['validation_split'])
        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer(f'hidden{i + 1}').set_weights(weights)

    train_model(saes, X_train, y_train, name, scats, config)

def main():
    os.makedirs(config['training']['model_save_path'], exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm", help="Model to train: lstm, gru, bilstm, cnnlstm, saes")
    parser.add_argument("--n_scats", type=int, default=None, help="Number of SCATS to train (5, 10, or None for all)")
    args = parser.parse_args()

    lag = config['data']['lag']
    file_path = config['data']['file_path']
    df_melted, scaler, street_segments = process_data(file_path, lag, n_scats=args.n_scats)
    data_dict = prepare_model_data(df_melted, sequence_length=lag)

    for scats, data in data_dict.items():
        X_train, y_train = data['X_train'], data['y_train']

        if args.model == 'lstm':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            m = model.get_lstm([lag, 64, 64, 1])
        elif args.model == 'gru':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            m = model.get_gru([lag, 64, 64, 1])
        elif args.model == 'bilstm':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            m = model.get_bidirectional_lstm([lag, 64, 64, 1])
        elif args.model == 'cnnlstm':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            m = model.get_cnn_lstm([lag, 64, 64, 1], n_steps=lag, n_features=1)
        elif args.model == 'saes':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
            m = model.get_saes([lag, 400, 400, 400, 1])
            train_seas(m, X_train, y_train, args.model, scats, config)
            continue
        else:
            print(f"Unknown model: {args.model}")
            return

        train_model(m, X_train, y_train, args.model, scats, config)

if __name__ == '__main__':
    main()