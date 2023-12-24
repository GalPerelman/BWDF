import os
import pandas as pd
import numpy as np
import random
import datetime
import argparse
import itertools
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

import constants
import utils
import evaluation
import multi_series
from data_loader import Loader
from preprocess import Preprocess
from forecast import Forecast
from lstm_model import LSTMForecaster
from prophet_model import ProphetForecaster
from params_grids import grids


def get_metrics(data, pred, horizon):
    observed, predicted = utils.get_dfs_commons(data, pred)
    i1, i2, i3, mape = None, None, None, None
    if horizon == 'short':
        i1 = evaluation.mean_abs_error(observed.iloc[:24], predicted.iloc[:24])
        i2 = evaluation.max_abs_error(observed.iloc[:24], predicted.iloc[:24])
        mape = evaluation.mean_abs_percentage_error(observed.iloc[:24], predicted.iloc[:24])

    elif horizon == 'long':
        i3 = evaluation.mean_abs_error(observed.iloc[24:], predicted.iloc[24:])
        mape = evaluation.mean_abs_percentage_error(observed.iloc[24:], predicted.iloc[24:])

    return i1, i2, i3, mape


def predict_dma(data, dma_name, model_name, model_params, dates_idx, horizon, cols_to_lag, lag_target, cols_to_move_stat,
                window_size, cols_to_decompose, decompose_target, norm_method):

    models = {'xgb': xgb.XGBRegressor, 'rf': RandomForestRegressor, 'prophet': ProphetForecaster,
              'lstm': LSTMForecaster, 'multi': multi_series.MultiSeriesForecaster}

    def predict(data, dma_name, model_name, params, start_train, start_test, end_test,
                cols_to_lag, cols_to_move_stat, window_size, cols_to_decompose, norm_method):

        f = Forecast(data=data, y_label=dma_name, cols_to_lag=cols_to_lag, cols_to_move_stat=cols_to_move_stat,
                     window_size=window_size, cols_to_decompose=cols_to_decompose, norm_method=norm_method,
                     start_train=start_train, start_test=start_test, end_test=end_test)

        if model_name == 'lstm':
            return f.format_forecast(f.predict(model=LSTMForecaster, params=params))

        elif model_name == 'multi':
            return f.multi_series_predict(params=params)

        elif model_name in ['xgb', 'rf', 'prophet']:
            return f.one_step_loop_predict(model=models[model_name], params=params)

    dates = constants.EXPERIMENTS_DATES[dates_idx]
    start_train = dates['start_train']
    start_test = dates['start_test']
    end_test = start_test + datetime.timedelta(days=1)
    if horizon == 'long':
        end_test = start_test + datetime.timedelta(days=7)

    if decompose_target:
        _cols_to_decompose = cols_to_decompose + [dma_name]
    else:
        _cols_to_decompose = cols_to_decompose

    predictions = predict(data=data, dma_name=dma_name, model_name=model_name, params=model_params,
                          start_train=start_train, start_test=start_test, end_test=end_test,
                          cols_to_lag={**cols_to_lag, **{dma_name: lag_target}}, cols_to_move_stat=cols_to_move_stat,
                          window_size=window_size, cols_to_decompose=_cols_to_decompose, norm_method=norm_method)

    predictions.columns = [dma_name]
    i1, i2, i3, mape = get_metrics(data, predictions, horizon=horizon)

    result = pd.DataFrame({
        'dma': dma_name,
        'model_name': model_name,
        'model_params': [model_params],
        'dates_idx': dates_idx,
        'start_train': dates['start_train'],
        'start_test': dates['start_test'],
        'end_test': dates['end_test'],
        'horizon': horizon,
        'lags': [{**cols_to_lag, **{dma_name: lag_target}}],
        'cols_to_move_stat': [cols_to_move_stat],
        'window_size': window_size,
        'cols_to_decompose': [_cols_to_decompose],
        'norm': norm_method,
        'i1': i1,
        'i2': i2,
        'i3': i3,
        'mape': mape
    })
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do', type=str, required=True)
    parser.add_argument('--dma_idx', type=int, required=False)
    parser.add_argument('--model_name', type=str, required=False)
    parser.add_argument('--dates_idx', type=int, required=False)
    parser.add_argument('--horizon', type=str, required=False)
    parser.add_argument('--norm_method', type=str, required=False)
    parser.add_argument('--output_dir', type=str, required=False)
    args = parser.parse_args()

    if args.do == 'experiment':
        run_experiment(args)
    elif args.do == 'multi_series_tune':
        multi_series.tune_dma(constants.DMA_NAMES[args.dma_idx])

    return args


def list_elements_combinations(elements):
    """
    Generate possible combinations of columns - for decomposing, moving stat etc.
    The order of columns in each combination does not matter

    A combination of k items from a set of n elements (n >= k) is denoted as C(n, k).
    The function generates combinations for all possible values of k (from 1 to the length of the list)

    Example:
    generate_combinations(['a', 'b', 'c'])
    [['a'], ['b'], ['c'], ['a', 'b'], ['a', 'c'], ['b', 'c'], ['a', 'b', 'c']]
    """
    all_combinations = [[]]
    for r in range(1, len(elements) + 1):
        all_combinations.extend(itertools.combinations(elements, r))
    return [list(comb) for comb in all_combinations]


def generate_parameter_sets(param_grid):
    keys, values = zip(*param_grid.items())

    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def generate_filename(args):
    args_dict = vars(args)
    filename = "".join(f"--{key}-{value}" for key, value in args_dict.items() if key not in ['do', 'output_dir'])
    filename += '.csv'
    return filename


def run_experiment(args):
    loader = Loader()
    p = Preprocess(loader.inflow, loader.weather, cyclic_time_features=True, n_neighbors=3)
    data = p.data

    results = pd.DataFrame()
    output_dir = utils.validate_dir_path(args.output_dir)
    output_file = generate_filename(args)

    model_info = grids[args.model_name]
    params = model_info['params']

    weather_cols_combs = list_elements_combinations(constants.WEATHER_COLUMNS)

    for params in generate_parameter_sets(params):
        for rain_lags in [0, 6, 12]:
            for temp_lags in [0, 6, 12]:
                for wind_lags in [0, 6, 12]:
                    for humidity_lags in [0, 6, 12]:
                        for target_lags in [0, 12, 24]:
                            for cols_to_move_stats in weather_cols_combs:
                                for decompose_target in [True, False]:
                                    dma_name = constants.DMA_NAMES[args.dma_idx]
                                    cols_to_lag = {'Rainfall depth (mm)': rain_lags,
                                                   'Air temperature (°C)': temp_lags,
                                                   'Windspeed (km/h)': wind_lags,
                                                   'Air humidity (%)': humidity_lags,
                                                   }

                                    res = predict_dma(data=data,
                                                      dma_name=dma_name,
                                                      model_name=args.model_name,
                                                      model_params=params,
                                                      dates_idx=args.dates_idx,
                                                      horizon=args.horizon,
                                                      cols_to_lag=cols_to_lag,
                                                      lag_target=target_lags,
                                                      cols_to_move_stat=cols_to_move_stats,
                                                      window_size=168,
                                                      cols_to_decompose=[],
                                                      decompose_target=decompose_target,
                                                      norm_method=args.norm_method,
                                                      )

                                    results = pd.concat([results, res])
                                    results.to_csv(os.path.join(output_dir, output_file))


if __name__ == "__main__":
    global_seed = 42
    os.environ['PYTHONHASHSEED'] = str(global_seed)
    random.seed(global_seed)
    np.random.seed(global_seed)
    parse_args()