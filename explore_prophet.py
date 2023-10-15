import pandas as pd
import datetime
import matplotlib.pyplot as plt
from prophet import Prophet

import constants
import utils
import evaluation
from preprocess import Preprocess


def prophet_predict(data, dma_name, start_train, start_test, end_test):
    x_train, y_train, x_test, y_test = Preprocess.by_label(
        data=data, y_label=dma_name, n_lags=0, start_train=start_train, start_test=start_test, end_test=end_test
    )

    data = pd.merge(x_train, y_train, left_index=True, right_index=True)
    idx_name = data.index.name
    data.index = data.index.tz_localize(None)
    data = data.reset_index()
    data = data.rename(columns={idx_name: 'ds', dma_name: "y"})

    holidays = data.loc[data['is_special'] == 1, ['ds', 'is_special']]
    holidays.columns = ["ds", "holiday"]
    holidays['holiday'] = 'holiday'

    m = Prophet(holidays=holidays, changepoint_prior_scale=0.01)
    for col in constants.EXOG_COLUMNS:
        m.add_regressor(col)

    m.fit(data)
    future = m.make_future_dataframe(periods=168, freq='H')

    x_test.index = x_test.index.tz_localize(None)
    x_test = x_test.reset_index()
    x_test = x_test.rename(columns={idx_name: 'ds'})

    future = pd.merge(future, x_test, left_on='ds', right_on='ds')
    forecast = m.predict(future)
    return forecast['yhat']


data = utils.import_preprocessed("resources/preprocessed_data.csv")
start_train = data.index.min()
start_pred = constants.TEST_TIMES['w1']['start'] - datetime.timedelta(days=7)
end_short_pred = start_pred + datetime.timedelta(hours=24)
end_long_pred = start_pred + datetime.timedelta(hours=168)

fig, axes = plt.subplots(nrows=len(constants.DMA_NAMES), sharex=True, figsize=(12, 9))
for i, dma in enumerate(constants.DMA_NAMES):
    y = data.loc[(data.index >= start_pred) & (data.index < constants.TEST_TIMES['w1']['start']), dma]

    short_pred = prophet_predict(data=data, dma_name=dma, start_train=start_train,
                                 start_test=start_pred, end_test=end_short_pred)

    long_pred = prophet_predict(data=data, dma_name=dma, start_train=start_train,
                                start_test=start_pred, end_test=end_long_pred)

    pred = pd.concat([short_pred, long_pred.iloc[24:]])
    mae = evaluation.mean_abs_error(y, pred)
    axes[i].plot(y.index, y)
    axes[i].plot(y.index, pred)
    axes[i].text(0.015, 0.88, f"MAE={mae:.3f}", transform=axes[i].transAxes, ha='left', va='center')
    axes[i].grid()
    axes[i].set_ylabel(dma[:-6])

plt.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.9, hspace=0.1)
plt.show()
