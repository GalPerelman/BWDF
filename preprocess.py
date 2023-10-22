import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import constants


class Preprocess:
    def __init__(self, inflow: pd.DataFrame, weather: pd.DataFrame, cyclic_time_features: bool, n_neighbors: int):
        self.inflow = inflow
        self.weather = weather
        self.cyclic_time_features = cyclic_time_features
        self.n_neighbors = n_neighbors

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
        if not self.cyclic_time_features:
            self.data['month'] = self.data.index.month
            self.data['day'] = self.data.index.day
            self.data['hour'] = self.data.index.hour
            # self.data['weekday'] = self.data.index.day_name()
            self.data['weekday_int'] = (self.data.index.weekday + 1) % 7 + 1
            self.data['week_num'] = self.data.index.strftime('%U').astype(int) + 1

        elif self.cyclic_time_features:
            self.data['hour_sin'] = np.sin(self.data.index.hour * (2. * np.pi / 24))
            self.data['hour_cos'] = np.cos(self.data.index.hour * (2. * np.pi / 24))
            self.data['day_sin'] = np.sin(self.data.index.day * (2. * np.pi / 31))  # Assuming max 31 days in a month
            self.data['day_cos'] = np.cos(self.data.index.day * (2. * np.pi / 31))
            self.data['weekday_sin'] = np.sin(((self.data.index.weekday + 1) % 7 + 1) * (2. * np.pi / 7))
            self.data['weekday_cos'] = np.cos(((self.data.index.weekday + 1) % 7 + 1) * (2. * np.pi / 7))
            self.data['month_sin'] = np.sin(self.data.index.month * (2. * np.pi / 12))
            self.data['month_cos'] = np.cos(self.data.index.month * (2. * np.pi / 12))
            self.data['weeknum_sin'] = np.sin((self.data.index.strftime('%U').astype(int) + 1) * (2. * np.pi / 52))
            self.data['weeknum_cos'] = np.cos((self.data.index.strftime('%U').astype(int) + 1) * (2. * np.pi / 52))

        def is_dst(dt):
            return dt.dst() != pd.Timedelta(0)

        self.data['is_dst'] = self.data.index.map(is_dst).astype(int)
        self.data['is_special'] = self.data.index.normalize().isin(constants.SPECIAL_DATES).astype(int)

    @staticmethod
    def construct_lag_features(data: pd.DataFrame, labels: list, n_lags: int):
        lagged_cols = []
        for i in range(1, n_lags+1):
            for label in labels:
                data[label + f'_{i}'] = data[label].shift(i)
                lagged_cols.append(label + f'_{i}')

        # drop the n_lags first rows to clear Nans
        data = data.iloc[n_lags:]
        return data, lagged_cols

    @staticmethod
    def split_data(data, y_label, start_train, start_test, end_test):
        x_columns = list(data.columns)
        x_columns = list(set(x_columns) - set(constants.DMA_NAMES))

        x_train = data.loc[(data.index >= start_train) & (data.index < start_test), x_columns]
        y_train = data.loc[(data.index >= start_train) & (data.index < start_test), y_label]

        x_test = data.loc[(data.index >= start_test) & (data.index < end_test), x_columns]
        y_test = data.loc[(data.index >= start_test) & (data.index < end_test), y_label]
        return x_train, y_train, x_test, y_test

    @staticmethod
    def fit_transform(data, columns, method='standard'):
        scalers = {}
        for col in columns:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'min_max':
                scaler = MinMaxScaler(feature_range=(0, 1))

            data.loc[:, col] = scaler.fit_transform(data[[col]])
            scalers[col] = scaler

        return data, scalers

    @staticmethod
    def transform(data,  columns, scalers):
        for col in columns:
            scaler = scalers[col]
            data.loc[:, col] = scaler.transform(data[[col]])

        return data

    @staticmethod
    def drop_other_dmas(data, y_label):
        cols_to_drop = list(set(constants.DMA_NAMES) - set([y_label]))
        data = data.drop(cols_to_drop, axis=1)
        return data

    def export(self, path):
        self.data.to_csv(path)


