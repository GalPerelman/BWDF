import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsforecast import StatsForecast
from statsmodels.tsa.arima.model import ARIMA
from statsforecast.models import MSTL, AutoARIMA, SeasonalNaive, Theta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.base import BaseEstimator, RegressorMixin
from statsforecast.models import HoltWinters, CrostonClassic as Croston, HistoricAverage, DynamicOptimizedTheta as DOT
from statsforecast.models import SeasonalNaive, AutoARIMA, AutoTheta

import constants
import evaluation
import graphs
import utils
from preprocess import Preprocess


class StatsModels(BaseEstimator, RegressorMixin):
    def __init__(self, horizon=168, model_name='theta'):
        self.horizon = horizon
        self.model_name =model_name
        self.model = None

        self.future_exog_cols = []
        self.hist_exog_cols = []
        self.stat_exog_cols = []
        self.future_exog_data = None
        self.stat_exog_data = None
        self.train = None
        self.test = None
        self.models = None
        self.nf = None

    def to_nixtla(self, data):
        data.index = data.index.tz_localize(None)
        data = data.reset_index().rename(columns={data.index.name: 'ds'})
        exog_columns = [col for col in data.columns if col in self.future_exog_cols + ['ds']]
        value_cols = [col for col in data.columns if col in constants.DMA_NAMES]
        data = data.melt(id_vars=exog_columns, value_vars=value_cols, var_name='unique_id', value_name='y')
        return data

    def preprocess(self, x, y):
        self.stat_exog_data = constants.DMA_DATA.reset_index()
        self.stat_exog_data = self.stat_exog_data.rename(columns={'index': 'unique_id'})
        self.stat_exog_data = self.stat_exog_data.drop(['type'], axis=1)
        self.stat_exog_cols = [col for col in self.stat_exog_data if col != 'unique_id']
        self.train = self.to_nixtla(x)
        self.test = self.to_nixtla(y)

    def build_model(self):
        # these were selected after examine the statsmodels optional models
        optional_models = {
            'theta': Theta(season_length=24),
            'mstl': MSTL(season_length=[24, 24 * 7])
        }
        self.models = [optional_models[self.model_name]]

    def build_auto_models(self):
        self.models = [
            AutoARIMA(season_length=24),
            AutoTheta(season_length=24)
        ]

    def fit_predict(self, x, y):
        """
        For auto models only
        """
        self.preprocess(x, y)
        self.build_auto_models()
        future_data = self.test.loc[:, self.future_exog_cols + ['unique_id', 'ds']]
        fcst = StatsForecast(models=self.models, freq='H', fallback_model=SeasonalNaive(season_length=7),
                             n_jobs=-1)
        pred = fcst.forecast(df=self.train, h=self.horizon, X_df=future_data)
        pred = pred.reset_index()
        return pred

    def fit(self, x, y):
        self.preprocess(x, y)
        self.build_model()
        self.sf = StatsForecast(models=self.models, freq='H', verbose=False)
        self.sf.fit(df=self.train)

    def predict(self):
        future_data = self.test.loc[:, self.future_exog_cols + ['unique_id', 'ds']]
        # pred = self.sf.predict(h=self.horizon, X_df=future_data)
        pred = self.sf.forecast(df=self.train, h=self.horizon, X_df=future_data)
        pred = pred.reset_index()
        return pred

    def format_pred(self, pred, model_name):
        return pred.pivot_table(index='ds', columns='unique_id', values=model_name)

    def get_metrics(self, observed, pred):
        observed, predicted = utils.get_dfs_commons(observed, pred)
        i1, i2, i3, mape = None, None, None, None
        if self.horizon == 24:
            i1 = evaluation.mean_abs_error(observed.iloc[:24], predicted.iloc[:24])
            i2 = evaluation.max_abs_error(observed.iloc[:24], predicted.iloc[:24])
            mape = evaluation.mean_abs_percentage_error(observed.iloc[:24], predicted.iloc[:24])

        elif self.horizon == 168:
            i1 = evaluation.mean_abs_error(observed.iloc[:24], predicted.iloc[:24])
            i2 = evaluation.max_abs_error(observed.iloc[:24], predicted.iloc[:24])
            i3 = evaluation.mean_abs_error(observed.iloc[24:], predicted.iloc[24:])
            mape = evaluation.mean_abs_percentage_error(observed.iloc[24:], predicted.iloc[24:])

        return i1, i2, i3, mape

    def plot_pred(self, pred):
        dmas = list(self.test['unique_id'].unique())
        n_dmas = self.test['unique_id'].nunique()
        fig, axes = plt.subplots(nrows=n_dmas, ncols=len(self.models), sharex=True, figsize=(10, 8))
        axes = np.atleast_2d(np.array(axes))
        for i, dma in enumerate(dmas):
            observed = self.test.loc[self.test['unique_id'] == dma]
            observed = observed.rename(columns={'y': dma})
            observed = observed.set_index('ds')
            for j, model in enumerate(self.models):
                temp = pred.loc[pred['unique_id'] == dma]
                temp = temp.rename(columns={model.__class__.__name__: dma})
                temp = temp.set_index('ds')
                _observed, _predicted = utils.get_dfs_commons(observed[[dma]], temp[[dma]])
                axes[i, j] = graphs.plot_test(_observed, _predicted, ax=axes[i, j])
                if i == 0:
                    axes[i, j].set_title(model.__class__.__name__)

                if j == 0:
                    axes[i, j].set_ylabel(dma[:5])

        fig.align_ylabels()
        plt.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.9, hspace=0.2)


class SARIMAWrap(BaseEstimator, RegressorMixin):
    def __init__(self, p=0, d=1, q=2, P=0, D=1, Q=1, m=24):
        self.order = (p, d, q)
        self.seasonal_order = (P, D, Q, m)
        self.model = None

    def fit(self, X, y):
        self.model = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order).fit(disp=False)
        return self

    def predict(self, X):
        return self.model.forecast(steps=len(X))

    def set_params(self, **params):
        self.order = params.get('order', self.order)
        self.seasonal_order = params.get('seasonal_order', self.seasonal_order)
        return self

    def get_params(self, deep=True):
        return {'order': self.order, 'seasonal_order': self.seasonal_order}


if __name__ == "__main__":
    data = utils.import_preprocessed("resources/preprocessed_not_cyclic.csv")
    data = data.drop([col for col in constants.DMA_NAMES if col not in constants.DMA_NAMES], axis=1)

    horizon = 24
    dates = constants.DATES_OF_LATEST_WEEK
    train, test = Preprocess.train_test_split(data, dates['start_train'],
                                              dates['start_test'],
                                              dates['start_test'] + datetime.timedelta(hours=horizon))

    ar = StatsModels(horizon=horizon)
    ar.fit(x=train, y=test)
    pred = ar.predict()
    for model in ar.models:
        _pred = ar.format_pred(pred, model_name=model.__class__.__name__)
        i1, i2, i3, mape = ar.get_metrics(test, _pred)
        print(pd.DataFrame({'i1': i1, 'i2': i2, 'i3': i3, 'mape': mape}, index=constants.DMA_NAMES))

    plt.show()
