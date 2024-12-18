import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import L1L2
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.optimizers import Adam, RMSprop

import constants
import evaluation
import utils
import graphs
from preprocess import Preprocess
from data_loader import Loader


class LSTMForecaster(BaseEstimator, RegressorMixin):
    def __init__(self, look_back=24, epochs=10, batch_size=24, num_layers=2, units=100, dropout=0.2,
                 recurrent_dropout=0.2, optimizer='adam', learning_rate=0.001, activation='tanh'):

        self.look_back = look_back
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.units = units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.activation = activation

        self.model = None

    def max_abs_error(self, y_true, y_pred):
        """
        This function is identical to the one in 'evaluation' module
        The only differences is that it uses keras syntax to incorporate with the keras model training
        """
        return K.max(K.abs(y_true - y_pred))

    def build_model(self, n_features):
        self.model = Sequential()
        for i in range(self.num_layers):
            return_sequences = True if i < self.num_layers - 1 else False
            if i == 0:
                self.model.add(
                    LSTM(self.units, input_shape=(self.look_back, n_features),
                         return_sequences=return_sequences,
                         dropout=self.dropout,
                         recurrent_dropout=self.recurrent_dropout,
                         activation=self.activation
                         )
                )
            else:
                self.model.add(LSTM(self.units,
                               return_sequences=return_sequences,
                               dropout=self.dropout,
                               recurrent_dropout=self.recurrent_dropout,
                               activation=self.activation
                                    )
                               )

        self.model.add(Dense(self.units, activation=self.activation))
        self.model.add(Dense(1))

        # Compile the model
        if self.optimizer == 'adam':
            opt = Adam(learning_rate=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            opt = RMSprop(learning_rate=self.learning_rate)
        # Include other optimizers as needed
        self.model.compile(optimizer=opt,
                           loss='mean_absolute_error',
                           metrics=['mean_absolute_error', self.max_abs_error])

    # simplified model build
    # self.model.add(LSTM(units=self.units, input_shape=(self.look_back, n_features), return_sequences=True))
    # self.model.add(LSTM(units=self.units, return_sequences=False))
    # self.model.add(Dropout(self.dropout))
    # self.model.add(Dense(1))
    # self.model.compile(optimizer='adam', loss='mse')

    def fit(self, x, y, verbose=0):
        self.build_model(n_features=x.shape[1])
        train_size = int(len(x) * 0.8)

        # last `look_back` periods of the train data are saved to use in the predict method
        self.train_tail = x.iloc[-self.look_back:]
        x_train, x_valid = x.iloc[:train_size].values, x.iloc[train_size:].values
        y_train, y_valid = y.iloc[:train_size].values, y.iloc[train_size:].values

        train_gen = TimeseriesGenerator(x_train, y_train, length=self.look_back, batch_size=self.batch_size)
        valid_gen = TimeseriesGenerator(x_valid, y_valid, length=self.look_back, batch_size=self.batch_size)
        early_stopping_monitor = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
                                               baseline=None, restore_best_weights=True)
        history = self.model.fit(train_gen, epochs=self.epochs, batch_size=self.batch_size, validation_data=valid_gen,
                                 callbacks=[early_stopping_monitor], verbose=verbose)

        return history

    def validation_predict(self, x, y):
        self.build_model(n_features=x.shape[1])
        train_size = int(len(x) * 0.8)
        x_train, x_valid = x.iloc[:train_size], x.iloc[train_size:]
        y_train, y_valid = y.iloc[:train_size], y.iloc[train_size:]
        train_gen = TimeseriesGenerator(x_train, y_train, length=self.look_back, batch_size=self.batch_size)
        valid_gen = TimeseriesGenerator(x_valid, y_valid, length=self.look_back, batch_size=self.batch_size)

        pred = self.model.predict(valid_gen)
        return pred

    def future_predict_preprocess(self, n_periods):
        # generate data set that includes train and validation
        final_data = self.data.loc[self.data.index < self.start_test]

        # normalize the final data
        final_data = Preprocess.transform(final_data, columns=self.standard_cols, scalers=self.scalers)

        # split to features and target
        final_x = final_data.loc[:, [col for col in final_data.columns if col != self.y_label]]
        final_y = final_data.loc[:, self.y_label]
        final_train_x = []
        final_train_y = []

        for i in range(self.look_back, len(final_data) - n_periods + 1):
            final_train_x.append(final_x.iloc[i - self.look_back:i])
            final_train_y.append(final_y.iloc[i + n_periods - 1:i + n_periods])

        final_train_x, final_train_y = np.array(final_train_x), np.array(final_train_y)

        # prepare the future periods exogenous data
        future_exog = self.data.loc[(self.data.index >= self.start_test) & (self.data.index < self.end_test)]
        future_exog = future_exog.drop(self.y_label, axis=1)
        future_exog = future_exog.values

        return final_train_x, final_train_y, future_exog

    def predict(self, x):
        # x is a pandas DataFrame contains the data features - exogenous and lagged target
        # x is extended to include the last `look_back` periods of the train data
        x = pd.concat([self.train_tail, x], axis=0)

        n_periods = len(x) - self.look_back
        future_exog = x.iloc[self.look_back:].values
        initial_sequence_x = x.iloc[:self.look_back].values

        forecast = []
        current_sequence_x = initial_sequence_x.copy()

        for i in range(n_periods):
            # Predict the next step using only the feature sequence
            prediction = self.model.predict(current_sequence_x[np.newaxis, :, :], verbose=0)[0, 0]
            forecast.append(prediction)

            # Update the feature sequence:
            # 1. Remove the oldest data point
            # 2. Append the latest feature data from future_exog
            current_sequence_x = np.vstack((current_sequence_x[1:], future_exog[i].reshape(1, -1)))

        return forecast


def predict_all_dmas(data, start_train, start_test, end_test, cols_to_lag, lag_target, norm_method):
    test_true = data.loc[(data.index >= start_test) & (data.index < end_test)]
    df = pd.DataFrame(index=pd.date_range(start=start_test, end=end_test - datetime.timedelta(hours=1), freq='1H'))

    fig, axes = plt.subplots(nrows=len(constants.DMA_NAMES), sharex=True, figsize=(12, 9))

    for i, dma in enumerate(constants.DMA_NAMES):
        # TARGET LAGS MUST BE 0 IN LSTM MODEL
        _cols_to_lag = {**cols_to_lag, **{dma: lag_target}}

        preprocessed = Preprocess.run(data=data.copy(deep=True),
                                      y_label=dma,
                                      start_train=start_train,
                                      start_test=start_test,
                                      end_test=end_test,
                                      cols_to_lag=_cols_to_lag,
                                      cols_to_move_stat=[],
                                      window_size=24,
                                      cols_to_decompose=[],
                                      norm_method=norm_method,
                                      labels_cluster=[])

        x_train, y_train, x_test, y_test, scalers, norm_cols, y_labels = preprocessed
        look_back = 24
        lstm = LSTMForecaster(look_back=look_back, epochs=10, batch_size=24)
        lstm.fit(x_train, y_train)

        pred = lstm.predict(x_test)
        pred = np.array(pred).reshape(-1, 1)
        pred = scalers[dma].inverse_transform(pred).flatten()

        period_dates = pd.date_range(start=start_test, end=end_test - datetime.timedelta(hours=1), freq='1H')
        forecast = pd.DataFrame({dma: pred}, index=period_dates)
        try:
            axes[i] = graphs.plot_test(test_true[[dma]], forecast, ax=axes[i])
        except ValueError:
            axes[i].plot(forecast.index, forecast[dma])
            axes[i].grid()
        axes[i].set_ylabel(dma[:5])

    fig.align_ylabels()
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.9, hspace=0.1)
    return df


if __name__ == "__main__":
    # usage example
    loader = Loader()
    preprocess = Preprocess(loader.inflow, loader.weather, cyclic_time_features=True, n_neighbors=3)
    data = preprocess.data

    norm_methods = ['standard', 'min_max', 'robust', 'power', 'quantile']

    start_train = constants.W1_LATEST_WEEK['start_train']
    start_test = constants.W1_LATEST_WEEK['start_test']
    end_test = start_test + datetime.timedelta(hours=168)
    cols_to_lag = {'Air humidity (%)': 12, 'Rainfall depth (mm)': 12, 'Air temperature (°C)': 12, 'Windspeed (km/h)': 12}
    predict_all_dmas(data, start_train=start_train, start_test=start_test, end_test=end_test, cols_to_lag=cols_to_lag,
                     lag_target=0, norm_method='power')
    plt.show()