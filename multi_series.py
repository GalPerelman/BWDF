import sys

import pandas as pd
import numpy as np
import datetime
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.model_selection_multiseries import grid_search_forecaster_multiseries

import constants
import graphs
import evaluation
import utils
from preprocess import Preprocess
from data_loader import Loader
from params_grids import grids

warnings.filterwarnings("ignore")

clusters = {
    'DMA A (L/s)': ['DMA J (L/s)'],
    'DMA B (L/s)': ['DMA C (L/s)'],
    'DMA C (L/s)': ['DMA B (L/s)', 'DMA E (L/s)', 'DMA G (L/s)'],
    'DMA D (L/s)': ['DMA E (L/s)', 'DMA G (L/s)', 'DMA H (L/s)'],
    'DMA E (L/s)': ['DMA D (L/s)', 'DMA H (L/s)'],  ###
    'DMA F (L/s)': ['DMA G (L/s)'],  ###
    'DMA G (L/s)': ['DMA D (L/s)', 'DMA E (L/s)', 'DMA H (L/s)'],
    'DMA H (L/s)': ['DMA D (L/s)', 'DMA E (L/s)', 'DMA G (L/s)', 'DMA J (L/s)'],
    'DMA I (L/s)': ['DMA J (L/s)'],
    'DMA J (L/s)': [],  ###
}


class MultiSeriesForecaster:
    def __init__(self, regressor, params, y_labels):
        self.regressor = regressor
        self.params = params
        self.y_labels = y_labels
        self.model = self.build_model()

    def build_model(self):
        return ForecasterAutoregMultiSeries(regressor=self.regressor(**self.params), lags=24)

    def fit(self, train_data: pd.DataFrame, train_exog: pd.DataFrame):
        self.model.fit(series=train_data, exog=train_exog)

    def predict(self, test_exog):
        return self.model.predict(steps=len(test_exog), levels=self.y_labels, exog=test_exog)

    def max_abs_error(self, y_true, y_pred):
        """
        This function is identical to the one in 'evaluation' module
        The only differences are:
         - the multi-series framework used here 'skforecast' requires specific arguments naming (y_true, y_pred)
         - the framework is based on numpy arrays and not pd.DataFrames as in 'evaluation' module
        """
        metric = np.max(np.abs(y_true - y_pred), axis=0)
        return metric

    def grid_search(self, data, dates, exog, dma, lags_grid, param_grid, steps, refit, initial_train_size):
        """
        https://joaquinamatrodrigo.github.io/skforecast/0.10.1/user_guides/backtesting.html

        steps (int) - Number of steps to predict.
        initial_train_size (int) - Number of samples in the initial train split.
        fixed_train_size (bool) - If True, train size doesn't increase but moves by `steps` in each iteration
        refit (bool, int) - Whether to re-fit the in each iteration. If int, train 'n' iterations.
        """

        results = grid_search_forecaster_multiseries(
            forecaster=self.model,
            series=data.loc[(data.index >= dates['start_train']) & (data.index < dates['end_test'])],
            lags_grid=lags_grid,
            param_grid=param_grid,
            steps=steps,
            metric=['mean_absolute_error', self.max_abs_error],
            initial_train_size=initial_train_size,
            fixed_train_size=True,
            levels=self.y_labels,
            exog=exog,
            refit=refit,
            return_best=False,
            verbose=False
        )

        if steps == 24:
            pred_term = 'short'
        elif steps == 168:
            pred_term = 'long'

        results.to_csv(f'{pred_term}_{dma[:5]}_multiseries.csv', index=False)


def predict_all_dma(dates, models, plot=False, record_score=False):
    results = pd.DataFrame()
    if plot:
        fig, axes = plt.subplots(nrows=len(constants.DMA_NAMES), sharex=True, figsize=(10, 8))
        fig.align_ylabels()
        plt.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.9, hspace=0.2)

    for i, dma in enumerate(constants.DMA_NAMES[:1]):
        y_labels = [dma] + clusters[dma]

        short_model_name = models[dma[:5]]['short']['model_name']
        short_model_params = models[dma[:5]]['short']['params']
        long_model_name = models[dma[:5]]['long']['model_name']
        long_model_params = models[dma[:5]]['long']['params']

        start_train = dates['start_train']
        start_test = dates['start_test']
        end_short_pred = start_test + datetime.timedelta(days=1)
        end_long_pred = start_test + datetime.timedelta(days=7)

        train_data = data.loc[(data.index >= start_train) & (data.index < start_test), y_labels]
        test_data = data.loc[(data.index >= start_test) & (data.index < end_test), y_labels]
        exog_columns = [col for col in data.columns if col not in constants.DMA_NAMES]
        train_exog = data.loc[(data.index >= start_train) & (data.index < start_test), exog_columns]

        if end_long_pred != dates['end_test']:
            raise "Problem with input dates"

        short_test_exog = data.loc[(data.index >= start_test) & (data.index < end_short_pred), exog_columns]
        f = MultiSeriesForecaster(regressor=xgb.XGBRegressor, params=short_model_params, y_labels=y_labels)
        f.fit(train_data=train_data, train_exog=train_exog)
        pred_short = f.predict(test_exog=short_test_exog)[[dma]]

        long_test_exog = data.loc[(data.index >= start_test) & (data.index < end_long_pred), exog_columns]
        f = MultiSeriesForecaster(regressor=xgb.XGBRegressor, params=long_model_params, y_labels=y_labels)
        f.fit(train_data=train_data, train_exog=train_exog)
        pred_long = f.predict(test_exog=long_test_exog)[[dma]]

        pred = pd.concat([pred_short, pred_long.iloc[24:]])

        if record_score:
            i1, i2, i3 = evaluation.one_week_score(observed=data, predicted=pred)
            utils.record_results(dma=dma, short_model_name=short_model_name, long_model_name=long_model_name,
                                 dates=dates, lags={}, pred_type='', score=[i1, i2, i3]
                                 )
        if plot:
            try:
                axes[i] = graphs.plot_test(observed=test_data[[dma]], predicted=pred, ylabel=dma[:5], ax=axes[i])
            except Exception as e:
                print(e)
                axes[i].plot(pred.index, pred[dma])
                axes[i].grid()
                axes[i].set_ylabel(f"{dma[:5]}")

        results = pd.concat([results, pred], axis=1)

    return results


def tune_dma(dma):
    loader = Loader()
    dates = constants.DATES_OF_LATEST_WEEK
    data = Preprocess(loader.inflow, loader.weather, cyclic_time_features=True, n_neighbors=3).data
    data.index.freq = 'H'

    cols_to_lag = {'Rainfall depth (mm)': 12, 'Air temperature (°C)': 12, 'Windspeed (km/h)': 12,
                   'Air humidity (%)': 12}
    data, lagged_cols = Preprocess.lag_features(data, cols_to_lag=cols_to_lag)

    y_labels = [dma] + clusters[dma]
    start_train = dates['start_train']
    start_test = dates['start_test']
    end_short_pred = start_test + datetime.timedelta(days=1)
    end_long_pred = start_test + datetime.timedelta(days=7)
    exog_columns = [col for col in data.columns if col not in constants.DMA_NAMES]

    short_data = data.loc[(data.index >= start_train) & (data.index < end_short_pred), y_labels]
    short_exog = data.loc[(data.index >= start_train) & (data.index < end_short_pred), exog_columns]
    f = MultiSeriesForecaster(regressor=xgb.XGBRegressor, params={}, y_labels=y_labels)
    f.grid_search(data=short_data, dates=dates, exog=short_exog, dma=dma, lags_grid=[6, 12],
                  param_grid=grids['xgb']['params'], steps=24, refit=3,
                  initial_train_size=len(short_data) - (24 * 3))

    long_data = data.loc[(data.index >= start_train) & (data.index < end_long_pred), y_labels]
    long_exog = data.loc[(data.index >= start_train) & (data.index < end_long_pred), exog_columns]
    f = MultiSeriesForecaster(regressor=xgb.XGBRegressor, params={}, y_labels=y_labels)
    f.grid_search(data=long_data, dates=dates, exog=long_exog, dma=dma, lags_grid=[6, 12],
                  param_grid=grids['xgb']['params'], steps=168, refit=3,
                  initial_train_size=len(long_data) - (168 * 3))


if __name__ == "__main__":
    loader = Loader()
    dates = constants.DATES_OF_LATEST_WEEK
    data = Preprocess(loader.inflow, loader.weather, cyclic_time_features=True, n_neighbors=3).data
    data.index.freq = 'H'

    cols_to_lag = {'Rainfall depth (mm)': 12, 'Air temperature (°C)': 12, 'Windspeed (km/h)': 12, 'Air humidity (%)': 12}
    start_train, start_test, end_test = dates['start_train'], dates['start_test'], dates['end_test']
    data, lagged_cols = Preprocess.lag_features(data, cols_to_lag=cols_to_lag)

    # xgb_params = utils.read_json("xgb_params.json")
    # predict_all_dma(dates=dates, models=xgb_params, plot=True, record_score=False)
    # plt.show()