import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST

import constants
import evaluation
import utils
from preprocess import Preprocess


class PatchTransformer(BaseEstimator, RegressorMixin):
    def __init__(self, horizon=168, input_size=1, batch_size=32, encoder_layers=3, n_heads=16, max_steps=100,
                 patch_len=16, learning_rate=0.0001, activation='gelu', scaler_type='standard'):

        self.horizon = horizon
        self.input_size = input_size*horizon
        self.batch_size = batch_size
        self.encoder_layers = encoder_layers
        self.n_heads = n_heads
        self.max_steps = max_steps
        self.patch_len = patch_len
        self.learning_rate = learning_rate
        self.activation = activation
        self.scaler_type = scaler_type

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
        self.future_exog_cols = [_ for _ in x if _ in constants.WEATHER_COLUMNS + constants.TIME_COLUMNS]
        self.stat_exog_data = constants.DMA_DATA.reset_index()
        self.stat_exog_data = self.stat_exog_data.rename(columns={'index': 'unique_id'})
        self.stat_exog_data = self.stat_exog_data.drop(['type'], axis=1)
        self.stat_exog_cols = [col for col in self.stat_exog_data if col != 'unique_id']
        self.train = self.to_nixtla(x)
        self.test = self.to_nixtla(y)

    def build_model(self):
        self.models = [
            PatchTST(
                h=self.horizon,
                input_size=self.input_size,
                batch_size=self.batch_size,
                encoder_layers=self.encoder_layers,
                n_heads=self.n_heads,
                max_steps=self.max_steps,
                patch_len=self.patch_len,
                learning_rate=self.learning_rate,
                activation=self.activation,
                scaler_type=self.scaler_type
                 ),
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
        return pred

    def format_pred(self, pred):
        return pred.pivot_table(index='ds', columns='unique_id', values='PatchTST')

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
    data = utils.import_preprocessed("resources/preprocessed_not_cyclic.csv")
    dates = constants.DATES_JULY
    horizon=168

    data, added_cols = Preprocess.construct_decomposed_features(data, columns=constants.DMA_NAMES)
    train, test = Preprocess.train_test_split(data, dates['start_train'],
                                              dates['start_test'], dates['start_test'] + datetime.timedelta(hours=horizon))
    nn = PatchTransformer(max_steps=5, horizon=horizon)
    nn.fit(x=train, y=test)
    pred = nn.predict()
    print(pred)
    pred = nn.format_pred(pred)
    i1, i2, i3, mape = nn.get_metrics(test, pred)
    print(pd.DataFrame({'i1': i1, 'i2': i2, 'i3': i3, 'mape': mape}, index=constants.DMA_NAMES))

    plt.plot(pred)
    plt.show()