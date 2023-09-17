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
        self.datetime_categorize()

    def data_completion(self, data):
        knn_impute = KNNImputer(n_neighbors=self.n_neighbors)
        data_imputed = knn_impute.fit_transform(data)
        data_imputed = pd.DataFrame(data_imputed, columns=data.columns, index=data.index)
        return data_imputed

    def datetime_categorize(self):
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

        self.data['is_dst'] = self.data.index.map(is_dst)
        self.data['is_special'] = self.data.index.normalize().isin(constants.SPECIAL_DATES)


