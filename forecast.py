import pandas as pd
import xgboost as xgb
from preprocess import Preprocess

xgb.set_config(verbosity=0)


class Forecast:
    def __init__(self, data, y_label, n_lags, start_train, start_test, end_test):
        self.data = data
        self.y_label = y_label
        self.n_lags = n_lags
        self.start_train = start_train
        self.start_test = start_test
        self.end_test = end_test

        self.x_train, self.y_train, self.x_test, self.y_test = Preprocess.by_label(data=self.data,
                                                                                   y_label=self.y_label,
                                                                                   n_lags=self.n_lags,
                                                                                   start_train=self.start_train,
                                                                                   start_test=self.start_test,
                                                                                   end_test=self.end_test)

    def predict(self, model, params):
        reg = model(**params)
        reg.fit(self.x_train, self.y_train)

        pred = pd.DataFrame(reg.predict(self.x_test), index=self.x_test.index)
        return pred
