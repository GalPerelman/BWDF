import pandas as pd
import xgboost as xgb
from preprocess import Preprocess

xgb.set_config(verbosity=0)


class Forecast:
    def __init__(self, data, y_label, n_lags, len_train, len_test):
        self.data = data
        self.y_label = y_label
        self.n_lags = n_lags
        self.len_train = len_train
        self.len_test = len_test

        self.x_train, self.y_train, self.x_test, self.y_test = Preprocess.preprocess_to_label(data=self.data,
                                                                                              y_label=self.y_label,
                                                                                              n_lags=self.n_lags,
                                                                                              len_train=self.len_train,
                                                                                              len_test=self.len_test)

    def predict(self, model, params):
        reg = model(**params)
        reg.fit(self.x_train, self.y_train)

        pred = pd.DataFrame(reg.predict(self.x_test), index=self.x_test.index)
        return pred
