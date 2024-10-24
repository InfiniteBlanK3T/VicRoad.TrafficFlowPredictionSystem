# Training Models

The `train_model.py` script is responsible for training machine learning models and saving their weights and loss history. The script uses the following functionalities:

## Training a Single Model

The `train_model` function trains a single model using the provided training data. The model is compiled with the Mean Squared Error (MSE) loss function and the RMSprop optimizer. The training history, including the loss and Mean Absolute Percentage Error (MAPE), is saved to a CSV file. The trained model and the loss history are saved in the directory specified by `config["training"]["model_save_path"]` with filenames that include the current timestamp.

## Models

The script supports training the following models:

1. **LSTM (Long Short-Term Memory)**
2. **GRU (Gated Recurrent Unit)**
3. **SAES (Stacked Auto-Encoders)**
4. **BiLSTM (Bidirectional LSTM)**
5. **CNN-LSTM (Convolutional Neural Network LSTM)**

### Model Definitions

#### LSTM

```python
def get_lstm(units):
    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(LSTM(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))
    return model
```
#### GRU

```python
def get_gru(units):
    model = Sequential()
    model.add(GRU(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(GRU(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))
    return model
```
#### SAES

```python
def get_saes(layers):
    sae1 = _get_sae(layers[0], layers[1], layers[-1])
    sae2 = _get_sae(layers[1], layers[2], layers[-1])
    sae3 = _get_sae(layers[2], layers[3], layers[-1])

    saes = Sequential()
    saes.add(Dense(layers[1], input_dim=layers[0], name='hidden1'))
    saes.add(Activation('sigmoid'))
    saes.add(Dense(layers[2], name='hidden2'))
    saes.add(Activation('sigmoid'))
    saes.add(Dense(layers[3], name='hidden3'))
    saes.add(Activation('sigmoid'))
    saes.add(Dropout(0.2))
    saes.add(Dense(layers[4], activation='sigmoid'))

    models = [sae1, sae2, sae3, saes]
    return models
```

BiLSTM

```python
def get_bidirectional_lstm(units):
    model = Sequential()
    model.add(Bidirectional(LSTM(units[1], return_sequences=True), input_shape=(units[0], 1)))
    model.add(Bidirectional(LSTM(units[2])))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='linear'))
    return model
```

CNN-LSTM

```python
def get_cnn_lstm(units, n_steps, n_features):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units[1], return_sequences=True))
    model.add(LSTM(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='linear'))
    return model
```

### Example Usage
To train the models, use the following command:

Ensure that your config.yml file is correctly set up with the appropriate paths for saving the trained models and loss history.

### Configuration
The config.yml file should include the following paths:
```bash
data:
  file_path: 'Scats-Data-October-2006-Bundoora.csv'
  lag: 12
  n_scats_options: [5, 10, 'all']

model:
  batch_size: 256
  epochs: 3
  validation_split: 0.05

training:
  models: ['lstm', 'gru', 'bilstm', 'cnnlstm', 'saes']
  model_save_path: 'main/model/trained'
```