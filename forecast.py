import pandas as pd
import xgboost as xgb
from preprocess import Preprocess

import utils
from preprocess import Preprocess

class Forecast:
    def __init__(self, data, y_label, n_lags, start_train, start_test, end_test):
        self.data = data
        self.y_label = y_label
        self.n_lags = n_lags
        self.start_train = start_train
        self.start_test = start_test
        self.end_test = end_test

        self.data = utils.drop_other_dmas(self.data, self.y_label)
        self.data, self.lagged_cols = Preprocess.construct_lag_features(self.data, [self.y_label], n_lags=self.n_lags)
        self.x_train, self.y_train, self.x_test, self.y_test = Preprocess.split_data(data=self.data,
                                                                                     y_label=self.y_label,
                                                                                     start_train=self.start_train,
                                                                                     start_test=self.start_test,
                                                                                     end_test=self.end_test)

    def predict(self, model, params):
        reg = model(**params)
        reg.fit(self.x_train, self.y_train)

        pred = pd.DataFrame(reg.predict(self.x_test), index=self.x_test.index)
        return pred

    def one_step_loop_predict(self, model, params, n_periods):
        """
        Function to predict with lagged features

        param n_periods:    int, number of periods to predict
        :return:
        """
        pred = pd.DataFrame()

        reg = model(**params)
        reg.fit(self.x_train, self.y_train)

        for i in range(n_periods):
            next_step_idx = self.x_test.index[i]
            for j in range(self.n_lags):
                self.x_test.loc[next_step_idx, self.y_label + f'_{j + 1}'] = self.y_train.iloc[-(j + 1)]

            pred_value = reg.predict(self.x_test.iloc[[i]])[0]
            pred.loc[next_step_idx, self.y_label] = pred_value
            self.y_train.loc[next_step_idx] = pred_value

        return pred
