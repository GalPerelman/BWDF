import datetime
import numpy as np
import pandas as pd
import xgboost as xgb

import constants
import multi_series

import utils
from preprocess import Preprocess


class Forecast:
    def __init__(self, data, y_label, cols_to_lag, cols_to_move_stat, window_size, cols_to_decompose, norm_method,
                 start_train, start_test, end_test):

        self.data = data
        self.y_label = y_label
        self.cols_to_lag = cols_to_lag
        self.cols_to_move_stat = cols_to_move_stat
        self.window_size = window_size
        self.cols_to_decompose = cols_to_decompose
        self.norm_method = norm_method
        self.start_train = start_train
        self.start_test = start_test
        self.end_test = end_test

        if not self.cols_to_lag:
            self.cols_to_lag = {self.y_label: 0}

        temp_data = data.copy(deep=True)
        temp_data = Preprocess.drop_other_dmas(temp_data, self.y_label)
        temp_data, lagged_cols = Preprocess.lag_features(temp_data, cols_to_lag=self.cols_to_lag)
        temp_data, stat_cols = Preprocess.construct_moving_features(temp_data, cols_to_move_stat, window_size)
        temp_data, decomposed_cols = Preprocess.construct_decomposed_features(temp_data, self.cols_to_decompose)

        first_no_nan_idx = temp_data.apply(pd.Series.first_valid_index).max()
        n_rows_to_drop = temp_data.index.get_loc(first_no_nan_idx)
        temp_data = Preprocess.drop_preprocess_nans(temp_data, n_rows=n_rows_to_drop)
        if self.norm_method:
            norm_cols = constants.WEATHER_COLUMNS + lagged_cols + stat_cols + [self.y_label]
        else:
            norm_cols = None

        preprocessed = Preprocess.split_data(data=temp_data, y_label=self.y_label, start_train=self.start_train,
                                             start_test=self.start_test, end_test=self.end_test,
                                             norm_method=self.norm_method, norm_cols=norm_cols
                                             )

        # unpack preprocessed
        self.x_train, self.y_train, self.x_test, self.y_test, self.scalers = preprocessed

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

        if self.scalers:
            pred = self.format_forecast(pred)
        return pred

    def multi_series_predict(self, params):
        temp_data = self.data.copy()
        temp_data.index.freq = 'H'
        temp_data, lagged_cols = Preprocess.lag_features(temp_data, cols_to_lag=self.cols_to_lag)

        y_labels = [self.y_label] + multi_series.clusters[self.y_label]
        exog_columns = [col for col in temp_data.columns if col not in constants.DMA_NAMES]

        train = temp_data.loc[(temp_data.index >= self.start_train) & (temp_data.index < self.start_test)]
        test = temp_data.loc[(temp_data.index >= self.start_test) & (temp_data.index < self.end_test)]

        train_data = train[y_labels]
        test_data = test[y_labels]
        train_exog = train[exog_columns]
        test_exog = test[exog_columns]

        f = multi_series.MultiSeriesForecaster(regressor=xgb.XGBRegressor, params=params, y_labels=y_labels)
        f.fit(train_data=train_data, train_exog=train_exog)
        pred = f.predict(test_exog=test_exog)[[self.y_label]]
        return pred

    def format_forecast(self, pred):
        """
        Mainly for LSTM where the returned forecast is normalized array
        This function inverse normalize the results and forma it in a pandas df with datetime index
        """
        pred = np.array(pred).reshape(-1, 1)
        pred = self.scalers[self.y_label].inverse_transform(pred).flatten()
        period_dates = pd.date_range(start=self.start_test, end=self.end_test - datetime.timedelta(hours=1), freq='1H')
        forecast = pd.DataFrame({self.y_label: pred}, index=period_dates)
        return forecast