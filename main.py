import os
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from prophet_model import ProphetForecaster

import graphs
import evaluation
import warnings

from data_loader import Loader
from forecast import *

# warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def predict_dma(data, dma_name, model_name, params, start_train, start_test, end_test, cols_to_lag, norm_method,
                pred_type):

    f = Forecast(data=data, y_label=dma_name, cols_to_lag=cols_to_lag, norm_method=norm_method,
                 start_train=start_train, start_test=start_test, end_test=end_test)

    models = {'xgb': xgb.XGBRegressor, 'rf': RandomForestRegressor, 'prophet': ProphetForecaster,
              'lstm': LSTMForecaster}

    if model_name == 'lstm':
        pred = f.format_forecast(f.predict(model=LSTMForecaster, params=params))

    if model_name in ['xgb', 'rf', 'prophet']:
        if pred_type == 'multi-step':
            pred = f.predict(model=models[model_name], params=params)
        elif pred_type == 'step-ahead':
            pred = f.one_step_loop_predict(model=models[model_name], params=params)

    return pred


def predict_all_dmas(data, models: dict, dates: dict, cols_to_lag: dict, lag_target: int, norm_method: str,
                     prediction_type: str, record_score=True, plot=False):

    results = pd.DataFrame()
    if plot:
        fig, axes = plt.subplots(nrows=len(constants.DMA_NAMES), sharex=True, figsize=(10, 8))
        fig.align_ylabels()
        plt.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.9, hspace=0.2)

    for i, dma in enumerate(constants.DMA_NAMES):
        short_model_name = models[dma[:5]]['short']['model_name']
        short_model_params = models[dma[:5]]['short']['params']

        start_train = dates['start_train']
        start_test = dates['start_test']
        end_short_pred = start_test + datetime.timedelta(days=1)
        end_long_pred = start_test + datetime.timedelta(days=7)

        if end_long_pred != dates['end_test']:
            raise "Problem with input dates"

        pred_short = predict_dma(data=data, dma_name=dma, model_name=short_model_name, params=short_model_params,
                                 start_train=start_train, start_test=start_test, end_test=end_short_pred,
                                 cols_to_lag={**cols_to_lag, **{dma: lag_target}}, norm_method=norm_method,
                                 pred_type=prediction_type)

        long_model_name = models[dma[:5]]['long']['model_name']
        long_model_params = models[dma[:5]]['long']['params']
        pred_long = predict_dma(data=data, dma_name=dma, model_name=long_model_name, params=long_model_params,
                                start_train=start_train, start_test=start_test, end_test=end_long_pred,
                                cols_to_lag={**cols_to_lag, **{dma: lag_target}}, norm_method=norm_method,
                                pred_type=prediction_type)

        pred = pd.concat([pred_short, pred_long.iloc[24:]])
        pred.columns = [dma]

        if record_score:
            i1, i2, i3 = evaluation.one_week_score(observed=data, predicted=pred)
            utils.record_results(dma=dma, short_model_name=short_model_name, long_model_name=long_model_name,
                                 dates=dates, lags={**cols_to_lag, **{dma: lag_target}},
                                 pred_type=prediction_type, score=[i1, i2, i3]
                                 )

        if plot:
            try:
                y_true = data.loc[(data.index >= start_test) & (data.index < end_long_pred), dma]
                axes[i] = graphs.plot_test(observed=y_true, predicted=pred, ylabel=dma[:5], ax=axes[i])
            except Exception as e:
                print(e)
                axes[i].plot(pred.index, pred[dma])
                axes[i].grid()
                axes[i].set_ylabel(f"{dma[:5]}")

        results = pd.concat([results, pred], axis=1)

    return results


if __name__ == "__main__":
    global_seed = 42
    os.environ['PYTHONHASHSEED'] = str(global_seed)
    np.random.seed(global_seed)

    xgb_models = utils.read_json("xgb_params.json")
    prophet_models = utils.read_json("prophet_params.json")
    lstm_models = utils.read_json('lstm_params.json')

    loader = Loader()
    data = Preprocess(loader.inflow, loader.weather, cyclic_time_features=True, n_neighbors=3).data

    results = predict_all_dmas(
        data=data,
        models=lstm_models,
        dates=constants.DATES_OF_LATEST_WEEK,
        cols_to_lag={'Air humidity (%)': 6},
        lag_target=0,
        norm_method='standard',
        prediction_type="step-ahead",
        record_score=True,
        plot=True
                               )

    results = predict_all_dmas(
        data=data,
        models=prophet_models,
        dates=constants.DATES_OF_LATEST_WEEK,
        cols_to_lag={'Air humidity (%)': 6},
        lag_target=12,
        norm_method='',
        prediction_type="step-ahead",
        record_score=True,
        plot=True
    )

plt.show()
