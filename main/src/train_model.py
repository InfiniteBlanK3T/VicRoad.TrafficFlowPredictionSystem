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
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yml')

with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

def train_model(model, X_train, y_train, name, scats, config):
    """
    Train a single model and save its weights and loss history.
    
    Args:
        model (keras.Model): The model to train.
        X_train (np.array): Training input data.
        y_train (np.array): Training target data.
        name (str): Name of the model.
        scats (int): SCATS number.
        config (dict): Configuration dictionary.
    """
    try:
        model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
        hist = model.fit(
            X_train, y_train,
            batch_size=config['model']['batch_size'],
            epochs=config['model']['epochs'],
            validation_split=config['model']['validation_split'])

        model_save_path = f"{config['training']['model_save_path']}/{name}_{scats}.h5"
        model.save(model_save_path)
        pd.DataFrame.from_dict(hist.history).to_csv(f"{config['training']['model_save_path']}/{name}_{scats}_loss.csv",
                                                    encoding='utf-8', index=False)
        logger.info(f"Model {name} for SCATS {scats} trained and saved successfully.")
    except Exception as e:
        logger.error(f"Error training model {name} for SCATS {scats}: {str(e)}")
        raise

def train_seas(models, X_train, y_train, name, scats, config):
    """
    Train the SAEs model.
    
    Args:
        models (list): List of SAE models.
        X_train (np.array): Training input data.
        y_train (np.array): Training target data.
        name (str): Name of the model.
        scats (int): SCATS number.
        config (dict): Configuration dictionary.
    """
    try:
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
    except Exception as e:
        logger.error(f"Error training SAEs model for SCATS {scats}: {str(e)}")
        raise

def parse_n_scats(value):
    if value.lower() == 'all':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("n_scats must be 'all' or an integer")
    
def main():

    os.makedirs(config['training']['model_save_path'], exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm", help="Model to train: lstm, gru, bilstm, cnnlstm, saes")
    parser.add_argument("--n_scats", type=parse_n_scats, default=None, help="Number of SCATS to train ('all' or an integer)")
    args = parser.parse_args()

    lag = config['data']['lag']
    file_path = config['data']['file_path']
    
    try:
        # 1. Data preparation
        df_melted, scaler, street_segments = process_data(file_path, lag, n_scats=args.n_scats)
        data_dict = prepare_model_data(df_melted, sequence_length=lag)

        # 2. Model training
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
                logger.error(f"Unknown model: {args.model}")
                return

            train_model(m, X_train, y_train, args.model, scats, config)
    except Exception as e:
        logger.error(f"An error occurred during model training: {str(e)}")

if __name__ == '__main__':
    main()