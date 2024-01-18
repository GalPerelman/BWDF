import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, QuantileTransformer, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin

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
    def outliers_cleaning(data, method, z_threshold, iqr_param, window_size):
        """
        Detects outliers in a pandas DataFrame column and replaces them with NaN.

        """
        df = data.copy()
        non_negative_columns = constants.DMA_NAMES + ['Rainfall depth (mm)', 'Windspeed (km/h)', 'Air humidity (%)']
        non_negative_columns = list(set(non_negative_columns) & set(list(data.columns)))
        for i, col in enumerate(non_negative_columns):
            df.loc[df[col] < 0, col] = np.nan

        numeric_columns = df.select_dtypes(include=np.number).columns
        for i, col in enumerate(numeric_columns):
            if method == 'z_score':
                z_scores = stats.zscore(df[col])
                outliers = (z_scores > z_threshold) | (z_scores < -z_threshold)

            elif method == 'iqr':
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - iqr_param * iqr
                upper_bound = q3 + iqr_param * iqr
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)

            elif method == "rolling_iqr":
                rolling_q1 = data[col].rolling(window=window_size).quantile(0.25)
                rolling_q3 = data[col].rolling(window=window_size).quantile(0.75)

                # Calculate the rolling IQR
                rolling_iqr = rolling_q3 - rolling_q1
                lower_bound = rolling_q1 - (iqr_param * rolling_iqr)
                upper_bound = rolling_q3 + (iqr_param * rolling_iqr)
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)

            else:
                outliers = []

            df.loc[outliers, col] = np.nan

        return df

    @staticmethod
    def detect_stuck_data_consecutive(data, consecutive_threshold):
        """
        Detects stuck data in a time series based on consecutive values being the same.
        """
        df = data.copy()
        for col in df.columns:
            diff = df[col].diff().ne(0)
            groups = diff.cumsum()
            group_sizes = df.groupby(groups)[col].transform('size')
            outliers = group_sizes >= consecutive_threshold
            df.loc[outliers, col] = np.nan

        return df

    @staticmethod
    def construct_moving_features(data, columns, window_size):
        added_cols = []
        for col in columns:
            data[col + f'_mavg'] = data[col].rolling(window=window_size).mean()
            data[col + f'_mstd'] = data[col].rolling(window=window_size).std()
            added_cols += [col + f'_mavg', col + f'_mstd']

        return data, added_cols

    @staticmethod
    def construct_decomposed_features(data, columns, period=24):
        added_cols = []
        for col in columns:
            filled_col = data[col].fillna(method='ffill')
            decomposition = seasonal_decompose(filled_col, model='additive', period=period, two_sided=False)

            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid

            trend[data[col].isna()] = np.nan
            seasonal[data[col].isna()] = np.nan
            residual[data[col].isna()] = np.nan

            data[col + f'_trend'] = trend
            data[col + f'_seasonal'] = seasonal
            data[col + f'_residual'] = residual
            added_cols += [col + f'_trend', col + f'_seasonal', col + f'_residual']

        return data, added_cols

    @staticmethod
    def lag_features(data: pd.DataFrame, cols_to_lag: dict):
        lagged_cols = []
        for label, lags in cols_to_lag.items():
            for lag in range(1, lags + 1):
                data[label + f'_{lag}'] = data[label].shift(lag)
                lagged_cols.append(label + f'_{lag}')

        return data, lagged_cols

    @staticmethod
    def drop_preprocess_nans(data, n_rows):
        """
        Some preprocess functions result in rows with nan values in the first rows
        For example, lagging features generate Nans since there are no previous values that can be lagged
        Another example is the moving statistics features
        The function looks for nans only in the first n_rows
        In the data end there are nans where exogenous data exist and target values are missing
        """
        data = data.iloc[n_rows:]
        return data

    @staticmethod
    def train_test_split(data, start_train, start_test, end_test):
        train = data.loc[(data.index >= start_train) & (data.index < start_test)]
        test = data.loc[(data.index >= start_test) & (data.index < end_test)]
        return train, test

    @staticmethod
    def split_data(data, y_label, start_train, start_test, end_test, norm_method, norm_cols, norm_param):
        x_columns = list(data.columns)
        x_columns = list(set(x_columns) - set(constants.DMA_NAMES))

        train = data.loc[(data.index >= start_train) & (data.index < start_test)]
        test = data.loc[(data.index >= start_test) & (data.index < end_test)]
        if norm_method and norm_cols is not None:
            train, scalers = Preprocess.fit_transform(train, columns=norm_cols, method=norm_method, param=norm_param)
            test = Preprocess.transform(test, columns=norm_cols, scalers=scalers)
        else:
            scalers = None

        train = train.dropna()  # this is to make sure there are no nans
        x_train = train.loc[:, x_columns]
        y_train = train.loc[:, y_label]

        x_test = test.loc[:, x_columns]
        y_test = test.loc[:, y_label]

        return x_train, y_train, x_test, y_test, scalers

    @staticmethod
    def fit_transform(data, columns, method, param):
        scalers = {}
        for col in columns:
            if method == '':
                continue
            elif method == 'standard':
                scaler = StandardScaler()
            elif method == 'min_max':
                scaler = MinMaxScaler(feature_range=(0, 1))
            elif method == 'robust':
                scaler = RobustScaler()
            elif method == 'power':
                scaler = PowerTransformer()
            elif method == 'quantile':
                scaler = QuantileTransformer()
            elif method == 'moving_stat':
                scaler = MovingWindowScaler(window_size=param)
            elif method == 'fixed_window':
                scaler = FixedWindowScaler()
            elif method == 'diff':
                scaler = DifferencingScaler(method=method)

            data.loc[:, col] = scaler.fit_transform(data[[col]])
            scalers[col] = scaler

        return data, scalers

    @staticmethod
    def transform(data, columns, scalers):
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

    @staticmethod
    def run(data, y_label, start_train, start_test, end_test, cols_to_lag, cols_to_move_stat, window_size,
            cols_to_decompose, norm_method='', labels_cluster=None):

        data.index.freq = 'H'

        if not labels_cluster:
            labels_cluster = None

        # if single target drop other dmas (cannot be used in train since will not be available for future periods)
        if labels_cluster is None:
            data = Preprocess.drop_other_dmas(data, y_label)
            y_labels = [y_label]  # for uniformity with the multi_series case
        else:
            y_labels = [y_label] + labels_cluster

        # if multi target y_labels is a list of all predicted labels and if target is lagged lag all targets
        if labels_cluster is not None and y_label in cols_to_lag.keys():
            for label in labels_cluster:
                cols_to_lag[label] = cols_to_lag[y_label]

        data, lagged_cols = Preprocess.lag_features(data, cols_to_lag=cols_to_lag)
        data, stat_cols = Preprocess.construct_moving_features(data, cols_to_move_stat, window_size)
        data, decomposed_cols = Preprocess.construct_decomposed_features(data, cols_to_decompose)
        # target is not available in future periods - decomposed components are lagged with window size
        if y_label in cols_to_decompose:
            for col in [y_label + f'_trend', y_label + f'_seasonal', y_label + f'_residual']:
                data[col] = data[col].shift(window_size)

        # drop nans before scaling - scalers will not be able to handle nans
        first_no_nan_idx = data.apply(pd.Series.first_valid_index).max()
        n_rows_to_drop = data.index.get_loc(first_no_nan_idx)
        data = Preprocess.drop_preprocess_nans(data, n_rows=n_rows_to_drop)

        train, test = Preprocess.train_test_split(data, start_train, start_test, end_test)

        norm_cols = constants.WEATHER_COLUMNS + lagged_cols + stat_cols + decomposed_cols + y_labels
        p = window_size if norm_method == 'moving_stat' else None
        if norm_method:
            train, scalers = Preprocess.fit_transform(train, columns=norm_cols, method=norm_method, param=p)
            test = Preprocess.transform(test, columns=norm_cols, scalers=scalers)
        else:
            scalers = None

        x_columns = [col for col in data.columns if col not in constants.DMA_NAMES]
        x_train = train.loc[:, x_columns]
        y_train = train.loc[:, y_labels].squeeze()  # squeeze y to series if one dimensional - support forecast models
        x_test = test.loc[:, x_columns]
        y_test = test.loc[:, y_labels].squeeze()  # squeeze y to series if one dimensional - support forecast models
        return x_train, y_train, x_test, y_test, scalers, norm_cols, y_labels


class MovingWindowScaler(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=168):
        self.window_size = window_size
        self.window_stats_ = {}

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Compute the rolling window statistics
        rolling_windows = pd.DataFrame(X).rolling(window=self.window_size, min_periods=1)
        means = rolling_windows.mean().values
        stds = rolling_windows.std().values

        # Shift the statistics to align with the 'future' values
        shifted_means = np.roll(means, -self.window_size, axis=0)
        shifted_stds = np.roll(stds, -self.window_size, axis=0)

        # Handle the first few values where shifting results in NaNs (using first available stats)
        for i in range(self.window_size):
            shifted_means[i] = means[self.window_size]
            shifted_stds[i] = stds[self.window_size]

        # Store the shifted statistics
        self.window_stats_ = {'mean': shifted_means, 'std': shifted_stds}

        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Use the last window statistics for all the points in X
        # These are the statistics of the last complete window from the training data
        last_mean = self.window_stats_['mean'][-1]
        last_std = self.window_stats_['std'][-1]

        X_normalized = (X - last_mean) / last_std
        X_normalized = np.nan_to_num(X_normalized)  # Handle cases where std is zero
        return X_normalized

    def inverse_transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Use the last window statistics for all the points in X
        # These are the statistics of the last complete window from the training data
        last_mean = self.window_stats_['mean'][-1]
        last_std = self.window_stats_['std'][-1]

        X_reconstructed = X * last_std + last_mean
        return X_reconstructed

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X, y)


class FixedWindowScaler(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=168):
        self.window_size = window_size
        self.window_stats_ = []
        self.last_window_stats_ = {'mean': None, 'std': None}

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame with a datetime index.")

        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a datetime index.")

        if X.empty:
            raise ValueError("Input DataFrame is empty.")

        num_features = X.shape[1]
        data_len = len(X)

        # Initialize arrays to store mean and std for each week and feature
        means = np.zeros((data_len, num_features))
        stds = np.zeros((data_len, num_features))

        # Group by week and calculate mean and std for each week
        for feature_idx in range(num_features):
            feature_column = X.iloc[:, feature_idx]
            weekly_groups = feature_column.groupby([X.index.isocalendar().year, X.index.isocalendar().week])

            for (year, week), group in weekly_groups:
                window_mean = group.mean()
                window_std = group.std()

                week_mask = (X.index.isocalendar().year == year) & (X.index.isocalendar().week == week)
                indices = np.where(week_mask)[0]
                means[indices, feature_idx] = window_mean
                stds[indices, feature_idx] = window_std

        self.window_stats_ = {'mean': means, 'std': stds}
        self.last_window_stats_['mean'] = means[-self.window_size:]
        self.last_window_stats_['std'] = stds[-self.window_size:]

        # Make sure no zeros in the std - to avoid division by zero z = (x- mu) / std
        # This caused an issue mainly when to many lagged columns were used
        self.window_stats_['std'][self.window_stats_['std'] < 10 ** -6] = 10 ** - 6
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values

        # If the number of rows in X is less than or equal to the window size, use the last window stats
        if len(X) <= self.window_size:
            last_mean = np.nanmean(self.last_window_stats_['mean'], axis=0)
            last_std = np.nanmean(self.last_window_stats_['std'], axis=0)
            X_normalized = (X - last_mean) / last_std
        else:
            X_normalized = (X - self.window_stats_['mean']) / self.window_stats_['std']

        X_normalized = np.nan_to_num(X_normalized)  # Handle cases where std is zero
        return X_normalized

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values

        # If the number of rows in X is less than or equal to the window size, use the last window stats
        if len(X) <= self.window_size:
            last_mean = np.nanmean(self.last_window_stats_['mean'], axis=0)
            last_std = np.nanmean(self.last_window_stats_['std'], axis=0)
            X_reconstructed = X * last_std + last_mean
        else:
            X_reconstructed = X * self.window_stats_['std'] + self.window_stats_['mean']

        return X_reconstructed


class DifferencingScaler(BaseEstimator, TransformerMixin):
    """
    Scaling data based on differencing between sequence records
    The class provide 3 differencing methods: standard diff, relative_diff, and log_diff
    In this project the methods relative_diff, and log_diff are not used
    Since they are highly sensitive to zeros and negatives values

    https://stats.stackexchange.com/a/549967
    """
    def __init__(self, lag=1, method='diff'):
        self.lag = lag
        self.init_values = None
        self.last_train_val = None
        self.method = method
        self.is_fitted = False

    def fit(self, X, y=None):
        # Store the initial values needed for the inverse transformation
        self.init_values = X.iloc[:self.lag, :]
        # Store the last value of the training set
        self.last_train_val = X.iloc[-1]
        self.is_fitted = True
        return self

    def transform(self, X, y=None, is_train_data=True):
        if self.is_fitted and self.last_train_val is not None:
            # Concatenate the last train value with the test set if scaler is already fitted
            last_value_df = pd.DataFrame(self.last_train_val).T
            X = pd.concat([last_value_df, X], axis=0)

        if self.method == 'diff':
            X_transformed = X.diff(periods=self.lag).dropna()
        else:
            raise ValueError("Invalid method specified")

        return X_transformed

    def inverse_transform(self, X, y=None):
        if self.method == 'diff':
            # Inverse of standard differencing
            last_value_df = pd.DataFrame(self.last_train_val).T
            restored = np.concatenate([last_value_df.values, X], axis=0).cumsum(axis=0)
        elif self.method == 'relative_diff':
            # Inverse of relative differencing
            restored = (1 + pd.concat([self.last_train_val, pd.DataFrame(X)])).cumprod() * self.init_values.values[-1]
        elif self.method == 'log_diff':
            # Inverse of logarithmic differencing
            restored = np.exp(pd.concat([self.last_train_val, pd.DataFrame(X)]).cumsum()) * self.init_values.values[-1]
        else:
            raise ValueError("Invalid method specified")
        return restored[1:]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)