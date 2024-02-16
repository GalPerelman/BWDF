import pandas as pd
import datetime
import logging
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, RNN, StemGNN, TimesNet, MLP, GRU, NHITS

from clusters import clusters
import constants
import evaluation
import utils
from data_loader import Loader
from preprocess import Preprocess

optional_nn_models = {"RNN": RNN, "StemGNN": StemGNN, "TimesNet": TimesNet, "MLP": MLP, "GRU": GRU, "NHITS": NHITS}


class NNForecaster(BaseEstimator, RegressorMixin):
    def __init__(self, dma, dmas_cluster, model_name: str, params: dict):
        self.dma = dma
        self.dmas_cluster = dmas_cluster
        self.model_name = model_name
        self.params = params
        self.horizon = params['h']

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
        exog_columns = [col for col in data.columns if col not in constants.DMA_NAMES]
        value_cols = [col for col in data.columns if col in constants.DMA_NAMES]
        data = data.melt(id_vars=exog_columns, value_vars=value_cols, var_name='unique_id', value_name='y')
        return data

    def preprocess(self, x, y):
        if self.dmas_cluster is not None:
            cols_to_drop = [col for col in constants.DMA_NAMES if col not in [self.dma] + self.dmas_cluster]
            x = x.drop(cols_to_drop, axis=1)
            y = y.drop(cols_to_drop, axis=1)

        self.future_exog_cols = [_ for _ in x if _ in constants.WEATHER_COLUMNS + constants.TIME_COLUMNS]
        self.stat_exog_data = constants.DMA_DATA.reset_index()
        self.stat_exog_data = self.stat_exog_data.rename(columns={'index': 'unique_id'})
        self.stat_exog_data = self.stat_exog_data.drop(['type'], axis=1)
        self.stat_exog_cols = [col for col in self.stat_exog_data if col != 'unique_id']
        self.train = self.to_nixtla(x)
        self.test = self.to_nixtla(y)

    def build_model(self):
        model = optional_nn_models[self.model_name]
        self.models = [
            model(**self.params),
        ]

    def fit(self, x, y):
        self.preprocess(x, y)
        self.build_model()
        self.nf = NeuralForecast(models=self.models, freq='H')
        self.nf.fit(df=self.train, static_df=self.stat_exog_data, verbose=False)

    def predict(self):
        future_data = self.test.loc[:, self.future_exog_cols + ['unique_id', 'ds']]
        pred = self.nf.predict(futr_df=future_data)
        pred = pred.reset_index()
        pred = self.format_pred(pred)[[self.dma]].values
        return pred

    def format_pred(self, pred):
        return pred.pivot_table(index='ds', columns='unique_id', values=self.model_name)

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


if __name__ == "__main__":
    # example usage
    models_config = utils.read_json("models_config_test.json")
    loader = Loader()
    prep = Preprocess(loader.inflow, loader.weather, cyclic_time_features=True, n_neighbors=3, outliers_config=None)
    dma = constants.DMA_NAMES[0]

    data = prep.data
    dates = constants.W1_PREV_YEAR
    horizon = 168

    data, added_cols = Preprocess.construct_decomposed_features(data, columns=constants.DMA_NAMES)
    train, test = Preprocess.train_test_split(data, dates['start_train'], dates['start_test'],
                                              dates['start_test'] + datetime.timedelta(hours=horizon))
    nn = NNForecaster(dma=dma,
                      dmas_cluster=clusters[0][dma],
                      model_name="GRU",
                      params={"h": 168, "input_size": 10, "inference_input_size": -1, "encoder_n_layers": 2,
                              "encoder_hidden_size": 200, "encoder_activation": "tanh"})

    nn.fit(x=train, y=test)
    pred = nn.predict()
    pred = nn.format_pred(pred)
    print(pred)
    i1, i2, i3, mape = nn.get_metrics(test, pred)
    print(pd.DataFrame({'i1': i1, 'i2': i2, 'i3': i3, 'mape': mape}, index=constants.DMA_NAMES))

    plt.plot(pred)
    plt.show()
