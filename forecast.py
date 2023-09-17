import pandas as pd
import xgboost as xgb


class FirstDayModel:
    def __init__(self, data, y_label, len_test=24):
        self.data = data
        self.y_label = y_label
        self.len_test = len_test
        self.x_train, self.y_train, self.x_test, self.y_test = self.split_data()

    def split_data(self):
        x_columns = list(self.data.columns)
        x_columns.remove(self.y_label)

        x_train = self.data.iloc[:-self.len_test].loc[:, x_columns]
        y_train = self.data.iloc[:-self.len_test].loc[:, self.y_label]

        x_test = self.data.iloc[-self.len_test:].loc[:, x_columns]
        y_test = self.data.iloc[-self.len_test:].loc[:, self.y_label]
        return x_train, y_train, x_test, y_test

    def xgb(self):
        reg = xgb.XGBRegressor(n_estimators=1000)
        reg.fit(self.x_train, self.y_train, verbose=True)

        pred = pd.DataFrame(reg.predict(self.x_test), index=self.x_test.index)
        return pred

