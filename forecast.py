import datetime
import numpy as np
import pandas as pd
import xgboost as xgb

import constants
from preprocess import Preprocess
from lstm_model import LSTMForecaster

import utils
from preprocess import Preprocess


class Forecast:
    def __init__(self, data, y_label, cols_to_lag, norm_method, start_train, start_test, end_test):
        self.data = data
        self.y_label = y_label
        self.cols_to_lag = cols_to_lag
        self.norm_method = norm_method
        self.start_train = start_train
        self.start_test = start_test
        self.end_test = end_test

        if not self.cols_to_lag:
            self.cols_to_lag = {self.y_label: 0}

        temp_data = self.data.copy()
        temp_data = utils.drop_other_dmas(temp_data, self.y_label)
        temp_data, lagged_cols = Preprocess.lag_features(temp_data, cols_to_lag=self.cols_to_lag)
        if self.norm_method:
            standard_cols = constants.WEATHER_COLUMNS + lagged_cols + [self.y_label]
        else:
            standard_cols = None

        (self.x_train, self.y_train,
         self.x_test, self.y_test, self.scalers) = Preprocess.split_data(data=temp_data,
                                                                         y_label=self.y_label,
                                                                         start_train=self.start_train,
                                                                         start_test=self.start_test,
                                                                         end_test=self.end_test,
                                                                         norm_method=self.norm_method,
                                                                         standard_cols=standard_cols
                                                                         )

    def predict(self, model, params):
        reg = model(**params)
        reg.fit(self.x_train, self.y_train)
        pred = self.x_test.copy()
        pred[self.y_label] = reg.predict(self.x_test)
        pred = pred[[self.y_label]]
        return pred

    def one_step_loop_predict(self, model, params):
        """
        Function to predict with lagged features

        param n_periods:    int, number of periods to predict
        :return:
        """
        n_periods = utils.num_hours_between_timestamps(self.start_test, self.end_test)
        n_lags = self.cols_to_lag[self.y_label]
        pred = pd.DataFrame()

        reg = model(**params)
        reg.fit(self.x_train, self.y_train)

        for i in range(n_periods):
            next_step_idx = self.x_test.index[i]
            for j in range(n_lags):
                self.x_test.loc[next_step_idx, self.y_label + f'_{j + 1}'] = self.y_train.iloc[-(j + 1)]

            pred_value = reg.predict(self.x_test.iloc[[i]])[0]
            pred.loc[next_step_idx, self.y_label] = pred_value
            self.y_train.loc[next_step_idx] = pred_value

        return pred

    def format_forecast(self, pred):
        """
        Mainly for LSTM where the returned forecast is normalized array
        This function inverse normalize the results and forma it in a pandas df with datetime index
        """
        pred = np.array(pred).reshape(-1, 1)
        pred = self.scalers[self.y_label].inverse_transform(pred).flatten()
        forecast_period_dates = pd.date_range(start=self.start_test,
                                              end=self.end_test - datetime.timedelta(hours=1), freq='1H')
        forecast = pd.DataFrame({self.y_label: pred}, index=forecast_period_dates)
        return forecast