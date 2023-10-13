import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scalecast.Forecaster import Forecaster
from prophet import Prophet
from sklearn.base import BaseEstimator, RegressorMixin

import evaluation
import utils
import constants
from preprocess import Preprocess

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class ProphetForecaster(BaseEstimator, RegressorMixin):
    def __init__(self,
                 seasonality_mode='additive',
                 daily_seasonality=False,
                 weekly_seasonality=False,
                 seasonality_prior_scale=10.0,
                 holidays_prior_scale=10.0,
                 changepoint_prior_scale=0.05):
        self.seasonality_mode = seasonality_mode
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.changepoint_prior_scale = changepoint_prior_scale

        self.model = None

    def fit(self, x, y):
        data = pd.merge(x, y, left_index=True, right_index=True)
        idx_name = data.index.name
        data.index = data.index.tz_localize(None)
        data = data.reset_index()
        data = data.rename(columns={idx_name: 'ds', y.name: "y"})

        holidays = data.loc[data['is_special'] == 1, ['ds', 'is_special']]
        holidays.columns = ["ds", "holiday"]
        holidays['holiday'] = 'holiday'
        self.model = Prophet(holidays=holidays,
                             seasonality_mode=self.seasonality_mode,
                             daily_seasonality=self.daily_seasonality,
                             weekly_seasonality=self.weekly_seasonality,
                             seasonality_prior_scale=self.seasonality_prior_scale,
                             holidays_prior_scale=self.holidays_prior_scale,
                             changepoint_prior_scale=self.changepoint_prior_scale
                             )
        for col in constants.EXOG_COLUMNS:
            self.model.add_regressor(col)

        self.model.fit(data)
        return self

    def predict(self, x):
        idx_name = x.index.name
        x.index = x.index.tz_localize(None)
        x = x.reset_index()
        x = x.rename(columns={idx_name: 'ds'})
        forecast = self.model.predict(x)
        return forecast['yhat'].values


if __name__ == "__main__":
    # usage example
    data = utils.import_preprocessed("resources/preprocessed_data.csv")
    start_train = constants.DATES_OF_LATEST_WEEK['start_train']
    start_test = constants.DATES_OF_LATEST_WEEK['start_test']
    end_short_pred = constants.DATES_OF_LATEST_WEEK['start_test'] + datetime.timedelta(days=1)
    end_long_pred = constants.DATES_OF_LATEST_WEEK['end_test']

    p = ProphetForecaster()
    params = {'seasonality_mode': 'additive', 'daily_seasonality': True, 'weekly_seasonality': True,
              'seasonality_prior_scale': 1, 'holidays_prior_scale': 2, 'changepoint_prior_scale': 0.01,
              }

    x_train, y_train, x_test, y_test = Preprocess.by_label(data=data,
                                                           y_label=constants.DMA_NAMES[0],
                                                           n_lags=0,
                                                           start_train=start_train,
                                                           start_test=start_test,
                                                           end_test=end_short_pred
                                                           )

    p.fit(x_train, y_train)
    y = p.predict(x_test)

    plt.plot(y_test.index, y_test.values)
    plt.plot(y_test.index, y)
    plt.show()


