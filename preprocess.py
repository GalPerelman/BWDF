import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

import constants


class Preprocess:
    def __init__(self, inflow: pd.DataFrame, weather: pd.DataFrame, n_neighbors: int, weather_lags: int):
        self.inflow = inflow
        self.weather = weather
        self.n_neighbors = n_neighbors
        self.weather_lags = weather_lags

        self.inflow = self.data_completion(self.inflow)
        self.weather = self.data_completion(self.weather)

        # merge on weather such that data will include test periods
        self.data = pd.merge(self.inflow, self.weather, left_index=True, right_index=True, how="right")
        self.construct_datetime_features()

    def data_completion(self, data):
        knn_impute = KNNImputer(n_neighbors=self.n_neighbors)
        data_imputed = knn_impute.fit_transform(data)
        data_imputed = pd.DataFrame(data_imputed, columns=data.columns, index=data.index)
        return data_imputed

    def construct_datetime_features(self):
        """
        Function to add categories based on the datetime index
        Month, weekday, hour, special dates
        """
        # self.data['month'] = self.data.index.month
        # self.data['day'] = self.data.index.day
        # self.data['hour'] = self.data.index.hour
        # self.data['weekday'] = self.data.index.day_name()
        # self.data['weekday_int'] = (self.data.index.weekday + 1) % 7 + 1
        # self.data['week_num'] = self.data.index.strftime('%U').astype(int) + 1

        self.data['hour_sin'] = self.data.index.hour * (2. * np.pi / 24)
        self.data['hour_cos'] = self.data.index.hour * (2. * np.pi / 24)
        self.data['day_sin'] = self.data.index.day * (2. * np.pi / 31)  # Assuming max 31 days in a month
        self.data['day_cos'] = self.data.index.day * (2. * np.pi / 31)
        self.data['weekday_sin'] = ((self.data.index.weekday + 1) % 7 + 1) * (2. * np.pi / 7)
        self.data['weekday_cos'] = ((self.data.index.weekday + 1) % 7 + 1) * (2. * np.pi / 7)
        self.data['month_sin'] = self.data.index.month * (2. * np.pi / 12)
        self.data['month_cos'] = self.data.index.month * (2. * np.pi / 12)
        self.data['weeknum_sin'] = (self.data.index.strftime('%U').astype(int) + 1) * (2. * np.pi / 52)
        self.data['weeknum_cos'] = (self.data.index.strftime('%U').astype(int) + 1) * (2. * np.pi / 52)

        def is_dst(dt):
            return dt.dst() != pd.Timedelta(0)

        self.data['is_dst'] = self.data.index.map(is_dst).astype(int)
        self.data['is_special'] = self.data.index.normalize().isin(constants.SPECIAL_DATES).astype(int)

    @staticmethod
    def construct_lag_features(data: pd.DataFrame, labels: list, n_lags: int):
        for i in range(1, n_lags+1):
            for label in labels:
                data[label + f'_{i}'] = data[label].shift(i)

        # drop the n_lags first rows to clear Nans
        data = data.iloc[n_lags:]
        return data

    @staticmethod
    def split_data(data, y_label, start_train, start_test, end_test):
        x_columns = list(data.columns)
        x_columns = list(set(x_columns) - set(constants.DMA_NAMES))

        x_train = data.loc[(data.index >= start_train) & (data.index < start_test), x_columns]
        y_train = data.loc[(data.index >= start_train) & (data.index < start_test), y_label]

        x_test = data.loc[(data.index >= start_test) & (data.index < end_test), x_columns]
        y_test = data.loc[(data.index >= start_test) & (data.index < end_test), y_label]
        return x_train, y_train, x_test, y_test

    def export(self, path):
        self.data.to_csv(path)


