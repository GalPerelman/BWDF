import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

import constants
import utils
from ar_model import SARIMAWrap
from lstm_model import LSTMForecaster
from preprocess import Preprocess
from prophet_model import ProphetForecaster
from multi_series import MultiSeriesForecaster


class Forecast:
    def __init__(self, data, y_label, cols_to_lag, cols_to_move_stat, window_size, cols_to_decompose, norm_method,
                 start_train, start_test, end_test, labels_cluster):

        self.data = data
        self.y_label = y_label
        self.cols_to_lag = cols_to_lag
        self.cols_to_move_stat = cols_to_move_stat
        self.window_size = window_size
        self.cols_to_decompose = cols_to_decompose
        self.norm_method = norm_method
        self.start_train = start_train
        self.start_test = start_test
        self.end_test = end_test
        self.labels_cluster = labels_cluster

        preprocessed = Preprocess.run(data=self.data.copy(deep=True),
                                      y_label=self.y_label,
                                      start_train=self.start_train,
                                      start_test=self.start_test,
                                      end_test=self.end_test,
                                      cols_to_lag=self.cols_to_lag,
                                      cols_to_move_stat=self.cols_to_move_stat,
                                      window_size=self.window_size,
                                      cols_to_decompose=self.cols_to_decompose,
                                      norm_method=self.norm_method,
                                      labels_cluster=self.labels_cluster)

        self.x_train, self.y_train, self.x_test, self.y_test, self.scalers, self.norm_cols, self.y_labels = preprocessed

    def predict(self, model, params):
        reg = model(**params)
        reg.fit(self.x_train, self.y_train)
        pred = self.x_test.copy()
        pred[self.y_label] = reg.predict(self.x_test)
        pred = pred[[self.y_label]]
        return pred

    def one_step_loop_predict(self, model, params):
        """
        Function to predict with lagged features
        param n_periods: int, number of periods to predict
        :return:
        """
        n_periods = utils.num_hours_between_timestamps(self.start_test, self.end_test)
        n_lags = self.cols_to_lag[self.y_label]
        pred = pd.DataFrame()

        reg = model(**params)
        reg.fit(self.x_train, self.y_train)

        for i in range(n_periods):
            next_step_idx = self.x_test.index[i]
            for j in range(n_lags):
                self.x_test.loc[next_step_idx, self.y_label + f'_{j + 1}'] = self.y_train.iloc[-(j + 1)]

            pred_value = reg.predict(self.x_test.iloc[[i]])[0]
            pred.loc[next_step_idx, self.y_label] = pred_value
            self.y_train.loc[next_step_idx] = pred_value

        if self.scalers is not None:
            pred = self.format_forecast(pred)
        return pred

    def multi_series_predict(self, params):
        # multi_series requires y_train will be pd.DataFrame and not pd.Series
        # the arguments to fit and predict methods are passed to ensure this requirement
        f = MultiSeriesForecaster(regressor=xgb.XGBRegressor, params=params, y_labels=self.y_labels)
        f.fit(train_data=pd.DataFrame(self.y_train), train_exog=pd.DataFrame(self.x_train))
        pred = f.predict(test_exog=pd.DataFrame(self.x_test))[[self.y_label]]

        if self.norm_cols is not None:
            pred = self.format_forecast(pred)

        return pred

    def format_forecast(self, pred):
        """
        Mainly for LSTM where the returned forecast is normalized array
        This function inverse normalize the results and forma it in a pandas df with datetime index
        """
        pred = np.array(pred).reshape(-1, 1)
        pred = self.scalers[self.y_label].inverse_transform(pred).flatten()
        period_dates = pd.date_range(start=self.start_test, end=self.end_test - datetime.timedelta(hours=1), freq='1H')
        forecast = pd.DataFrame({self.y_label: pred}, index=period_dates)
        return forecast


def folding_forecast(data, dma_name, cols_to_lag, cols_to_move_stat, window_size, cols_to_decompose, norm_method,
                     start_train, start_test, labels_cluster, model, params, horizon=24, folds=7):

    pred = pd.DataFrame()
    _data = data.copy(deep=True)
    _data = data.loc[data.index < start_test + datetime.timedelta(hours=horizon * folds)]

    for fold in range(folds):
        end_test = start_test + datetime.timedelta(hours=horizon)
        f = Forecast(data=_data, y_label=dma_name, cols_to_lag=cols_to_lag, cols_to_move_stat=cols_to_move_stat,
                     window_size=window_size, cols_to_decompose=cols_to_decompose, norm_method=norm_method,
                     start_train=start_train, start_test=start_test, end_test=end_test,
                     labels_cluster=labels_cluster)

        fold_pred = f.one_step_loop_predict(model=model, params=params)
        pred = pd.concat([pred, fold_pred])
        _data.loc[(_data.index >= start_test) & (_data.index < end_test), dma_name] = fold_pred

        start_test = start_test + datetime.timedelta(hours=horizon)

    return pred


def predict_dma(data, dma_name, model_name, params, start_train, start_test, end_test, cols_to_lag,
                cols_to_move_stat, window_size, cols_to_decompose, norm_method, labels_cluster, pred_type):
    f = Forecast(data=data, y_label=dma_name, cols_to_lag=cols_to_lag,
                 cols_to_move_stat=cols_to_move_stat, window_size=window_size, cols_to_decompose=cols_to_decompose,
                 norm_method=norm_method, start_train=start_train, start_test=start_test, end_test=end_test,
                 labels_cluster=labels_cluster)

    models = {'xgb': xgb.XGBRegressor, 'rf': RandomForestRegressor, 'prophet': ProphetForecaster,
              'lstm': LSTMForecaster, 'multi': MultiSeriesForecaster, 'sarima': SARIMAWrap}

    if model_name == 'lstm':
        pred = f.format_forecast(f.predict(model=LSTMForecaster, params=params))

    if model_name == 'multi':
        pred = f.multi_series_predict(params=params)

    if model_name in ['xgb', 'rf', 'prophet', 'arima', 'sarima']:
        # only if target label is not lagged, found to be less recommended
        if pred_type == 'multi-step':
            pred = f.predict(model=models[model_name], params=params)
               
        elif pred_type == 'step-ahead':
            pred = f.one_step_loop_predict(model=models[model_name], params=params)

    return pred