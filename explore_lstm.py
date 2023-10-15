import os
import pandas as pd
import numpy as np
import datetime
import math
import json

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint

import constants
import utils
from preprocess import Preprocess

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns  # List of column names to encode

    def fit(self, X, y=None):
        # Nothing to fit in this encoder
        return self

    def transform(self, X, y=None):
        """
        Transforms columns of X using a cyclical encoding.
        """
        if not self.columns:
            raise ValueError("Specify columns in the constructor or before calling transform.")

        for col in self.columns:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in input dataframe.")

            if col == 'hour':
                X['hour_sin'] = np.sin(X[col] * (2. * np.pi / 24))
                X['hour_cos'] = np.cos(X[col] * (2. * np.pi / 24))
            elif col == 'day':
                X['day_sin'] = np.sin(X[col] * (2. * np.pi / 31))  # Assuming max 31 days in a month
                X['day_cos'] = np.cos(X[col] * (2. * np.pi / 31))
            elif col == 'weekday_int':
                X['weekday_sin'] = np.sin(X[col] * (2. * np.pi / 7))
                X['weekday_cos'] = np.cos(X[col] * (2. * np.pi / 7))
            elif col == 'month':
                # Subtracting 1 from month to get values between 0-11 for January to December
                X['month_sin'] = np.sin((X[col] - 1) * (2. * np.pi / 12))
                X['month_cos'] = np.cos((X[col] - 1) * (2. * np.pi / 12))
            elif col == 'week_num':
                X['weeknum_sin'] = np.sin((X[col]) * (2. * np.pi / 52))
                X['weeknum_sin'] = np.cos((X[col]) * (2. * np.pi / 52))
            else:
                raise ValueError(f"Column '{col}' is not supported for cyclical encoding.")

            # Optionally drop the original column to reduce multicollinearity
            X.drop(col, axis=1, inplace=True)

        return X


class LSTMLoader:
    def __init__(self, data, y_label, start_train, start_test, end_test, seq_len):
        self.data = data
        self.y_label = y_label
        self.start_train = start_train
        self.start_test = start_test
        self.end_test = end_test
        self.seq_len = seq_len

        self.x_columns = list(set(data.columns) - set(constants.DMA_NAMES))
        self.data = self.data[self.x_columns + [self.y_label]]

        self.data_train, self.data_test = self.train_test_split()
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)

        cycle_features = ['hour', 'day', 'weekday_int', 'week_num', 'month']
        noncycle_features = constants.WEATHER_COLUMNS
        binary_features = ['is_special', 'is_dst']
        self.transformers = [('cyclical', CyclicalEncoder(columns=cycle_features), cycle_features),
                             ('noncyclical_numerical', MinMaxScaler(), noncycle_features),
                             ('binary', 'passthrough', binary_features),
                             ('target', MinMaxScaler(), [self.y_label])
                             ]

        self.preprocessor = ColumnTransformer(self.transformers)

    def train_test_split(self):
        train = self.data.loc[(self.data.index >= self.start_train) & (self.data.index < self.start_test)]
        test = self.data.loc[(self.data.index >= self.start_test) & (self.data.index < self.end_test)]
        return train, test

    def normalize(self, x):
        x_processed = self.preprocessor.fit_transform(x)
        return x_processed

    def get_train_data(self, normalize=True):
        data_x = []
        data_y = []
        for i in range(self.len_train - self.seq_len):
            x, y = self._next_window(i, self.seq_len, normalize)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def get_test_data(self, normalize=True):
        data_x = []
        data_y = []
        for i in range(self.len_test - self.seq_len):
            window = self.data_train.iloc[i:i + seq_len]
            window = self.normalize(window) if normalize else window
            x = window[:, :-1]
            y = window[-1, [0]]
            data_x.append(x)
            data_y.append(y)

        return np.array(data_x), np.array(data_y)

    def _next_window(self, i, seq_len, normalize):
        window = self.data_train.iloc[i:i + seq_len]
        window = self.normalize(window) if normalize else window
        x = window[:, :-1]
        y = window[-1, [0]]
        return x, y

    def normalize_windows(self, window_data, single_window=False):
        normalized_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalized_window = []
            for col_i in range(window.shape[1]):
                normalized_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalized_window.append(normalized_col)
            normalized_window = np.array(
                normalized_window).T  # reshape and transpose array back into original multidimensional format
            normalized_data.append(normalized_window)
        return np.array(normalized_data)


class LSTMForecaster(BaseEstimator, RegressorMixin):
    def __init__(self, config):
        self.config = config
        self.model = Sequential()

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build(self):
        for layer in self.config['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = x.shape[1]
            input_dim = x.shape[2]

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=self.config['model']['loss'], optimizer=self.config['model']['optimizer'])

    def train(self, x, y, save_dir):
        e = self.config['training']['epochs']
        save_fname = os.path.join(save_dir,
                                  '%s-e%s.keras' % (datetime.datetime.now().strftime('%d%m%Y-%H%M%S'), str(e)))
        callbacks = [EarlyStopping(monitor='val_loss', patience=2),
                     ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
                     ]
        self.model.fit(x=x,
                       y=y,
                       epochs=e,
                       batch_size=self.config['training']['batch_size'],
                       callbacks=callbacks)
        self.model.save(save_fname)

    def predict(self, x_test, loader):
        pred = self.model.predict(x_test)
        y_scaler = loader.preprocessor.named_transformers_['target']
        pred = y_scaler.inverse_transform(pred)
        return pred


if __name__ == "__main__":
    """
    https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction/tree/master
    """
    lstm_config = json.load(open('lstm_config.json', 'r'))
    seq_len = lstm_config['data']['sequence_length']
    batch_size = lstm_config['training']['batch_size']

    data = utils.import_preprocessed("resources/preprocessed_data.csv")
    start_train = constants.TZ.localize(datetime.datetime(2022, 7, 1, 0, 0))
    start_test = constants.TZ.localize(datetime.datetime(2022, 7, 17, 0, 0))
    end_short_pred = constants.TZ.localize(datetime.datetime(2022, 7, 18, 0, 0))

    start_train = data.index.min()
    start_test = constants.DATES_OF_LATEST_WEEK['start_test']
    end_test = constants.DATES_OF_LATEST_WEEK['start_test'] + datetime.timedelta(hours=24)

    loader = LSTMLoader(data, y_label=constants.DMA_NAMES[0], start_train=start_train, start_test=start_test,
                        end_test=end_test, seq_len=seq_len)

    x, y = loader.get_train_data(normalize=True)
    steps_per_epoch = math.ceil((loader.len_train - seq_len) / batch_size)

    lstm_model = LSTMForecaster(lstm_config)
    lstm_model.build()
    lstm_model.train(x=x, y=y, save_dir=constants.ROOT_DIR)

    # lstm_model.load_model(filepath="15102023-161612-e2.keras")
    x_test, y_test = loader.get_test_data()

    predictions = lstm_model.predict(x_test, loader=loader)
    print(predictions)
