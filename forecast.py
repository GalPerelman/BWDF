import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR

from hyperparam_search import Searcher
import warnings
xgb.set_config(verbosity=0)
warnings.filterwarnings(action="ignore")


class Forecast:
    def __init__(self, data, y_label, models, len_train, len_test, name):
        self.data = data
        self.y_label = y_label
        self.models = models
        self.len_train = len_train
        self.len_test = len_test
        self.name = name

        self.x_train, self.y_train, self.x_test, self.y_test = self.split_data()

    def split_data(self):
        x_columns = list(self.data.columns)
        x_columns.remove(self.y_label)

        x_train = self.data.iloc[-(self.len_train + self.len_test): -self.len_test].loc[:, x_columns]
        y_train = self.data.iloc[-(self.len_train + self.len_test): -self.len_test].loc[:, self.y_label]

        x_test = self.data.iloc[-self.len_test:].loc[:, x_columns]
        y_test = self.data.iloc[-self.len_test:].loc[:, self.y_label]
        return x_train, y_train, x_test, y_test

    def xgb(self):
        reg = xgb.XGBRegressor(n_estimators=500,
                               bootstrap=False,
                               max_depth=7,
                               min_samples_leaf=2,
                               min_samples_split=3,
                               colsample_bytree=0.7,
                               learning_rate=0.2,
                               reg_alpha=0.5
                               )
        reg.fit(self.x_train, self.y_train, verbose=True)

        pred = pd.DataFrame(reg.predict(self.x_test), index=self.x_test.index)
        return pred

    def search(self):
        for model_name, model_info in self.models.items():
            searcher = Searcher(model_name, model_info, self.x_train, self.y_train, self.y_label, self.name)
            searcher.grid_search()