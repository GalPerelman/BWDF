import pandas as pd
import numpy as np
import datetime
import math
import json

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import L1L2
from keras.preprocessing.sequence import TimeseriesGenerator

import constants
from preprocess import Preprocess
from data_loader import Loader


class LSTMForecaster:
    def __init__(self, data, start_train, start_valid, start_test, end_test, y_label, look_back):
        self.data = data
        self.start_train = start_train
        self.start_valid = start_valid
        self.start_test = start_test
        self.end_test = end_test
        self.y_label = y_label
        self.look_back = look_back

        self.train_data = None
        self.valid_data = None
        self.scalers = None

        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None

        self.model = None

        self.preprocess()

    def preprocess(self):
        train = self.data.loc[(self.data.index >= self.start_train) & (self.data.index < self.start_valid)]
        valid = self.data.loc[(self.data.index >= self.start_valid) & (self.data.index < self.start_test)]

        standardize_columns = constants.WEATHER_COLUMNS + [self.y_label]
        self.train_data, self.scalers = Preprocess.standardize(train, columns=standardize_columns)
        self.valid_data = Preprocess.standardize(valid, columns=standardize_columns, scalers=self.scalers)

        self.x_train = train.loc[:, [col for col in train.columns if col != self.y_label]]
        self.y_train = train.loc[:, self.y_label]
        self.x_valid = valid.loc[:, [col for col in valid.columns if col != self.y_label]]
        self.y_valid = valid.loc[:, self.y_label]

    def build_supervised(self):
        train_generator = TimeseriesGenerator(self.x_train, self.y_train, length=self.look_back, batch_size=12)
        valid_generator = TimeseriesGenerator(self.x_valid, self.y_valid, length=self.look_back, batch_size=12)
        return train_generator, valid_generator

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(self.look_back, self.x_train.shape[1])))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self):
        train_gen, valid_gen = self.build_supervised()
        history = self.model.fit(train_gen, epochs=12, batch_size=48, validation_data=valid_gen, shuffle=False)
        return history


if __name__ == "__main__":
    # usage example
    loader = Loader()
    preprocess = Preprocess(loader.inflow, loader.weather, n_neighbors=3, weather_lags=12)
    data = preprocess.data

