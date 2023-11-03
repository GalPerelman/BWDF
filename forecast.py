import pandas as pd
import xgboost as xgb

import graphs
from preprocess import Preprocess

import utils
from preprocess import Preprocess


class Forecast:
    def __init__(self, data, y_label, cols_to_lag, start_train, start_test, end_test):
        self.data = data
        self.y_label = y_label
        self.cols_to_lag = cols_to_lag
        self.start_train = start_train
        self.start_test = start_test
        self.end_test = end_test

        if not self.cols_to_lag:
            self.cols_to_lag = {self.y_label: 0}

        self.data = utils.drop_other_dmas(self.data, self.y_label)
        self.data, self.lagged_cols = Preprocess.lag_features(self.data, cols_to_lag)
        self.x_train, self.y_train, self.x_test, self.y_test = Preprocess.split_data(data=self.data,
                                                                                     y_label=self.y_label,
                                                                                     start_train=self.start_train,
                                                                                     start_test=self.start_test,
                                                                                     end_test=self.end_test)

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
