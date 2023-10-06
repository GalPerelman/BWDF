import pandas as pd
from sklearn.impute import KNNImputer

import constants


class Preprocess:
    def __init__(self, inflow: pd.DataFrame, weather: pd.DataFrame, n_neighbors: int):
        self.inflow = inflow
        self.weather = weather
        self.n_neighbors = n_neighbors

        self.inflow = self.data_completion(self.inflow)
        self.weather = self.data_completion(self.weather)
        self.data = pd.merge(self.inflow, self.weather, left_index=True, right_index=True, how="left")

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
        self.data['month'] = self.data.index.month
        self.data['day'] = self.data.index.day
        self.data['hour'] = self.data.index.hour
        # self.data['weekday'] = self.data.index.day_name()
        self.data['weekday_int'] = (self.data.index.weekday + 1) % 7 + 1
        self.data['week_num'] = self.data.index.strftime('%U').astype(int) + 1

        def is_dst(dt):
            return dt.dst() != pd.Timedelta(0)

        self.data['is_dst'] = self.data.index.map(is_dst).astype(int)
        self.data['is_special'] = self.data.index.normalize().isin(constants.SPECIAL_DATES).astype(int)

    @staticmethod
    def construct_lag_features(data, y_label, n_lags):
        for i in range(1, n_lags+1):
            data[y_label + f'_{i}'] = data[y_label].shift(i)

        data.dropna(inplace=True)
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

    @staticmethod
    def by_label(data, y_label, n_lags, start_train, start_test, end_test):
        data = Preprocess.construct_lag_features(data=data, y_label=y_label, n_lags=n_lags)
        x_train, y_train, x_test, y_test = Preprocess.split_data(data=data,
                                                                 y_label=y_label,
                                                                 start_train=start_train,
                                                                 start_test=start_test,
                                                                 end_test=end_test)
        return x_train, y_train, x_test, y_test

    def export(self, path):
        self.data.to_csv(path)


