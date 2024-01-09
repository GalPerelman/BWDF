import os
import time

import pandas as pd
import numpy as np
import random
import datetime
import argparse
import itertools
import logging
import traceback
import gc
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

import constants
import utils
import evaluation
import multi_series
import hyperparam_search
from data_loader import Loader
from preprocess import Preprocess
from forecast import Forecast
from lstm_model import LSTMForecaster
from prophet_model import ProphetForecaster
from params_grids import grids
from clusters import clusters

from logger import Logger

slurm_job_id = os.environ.get('SLURM_JOB_ID')
if slurm_job_id is None:
    slurm_job_id = 'slurm_id'
logger = Logger(name=f'experiment_{slurm_job_id}', LOGGING_DIR='logging').get()
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").disabled = True
logging.getLogger('tensorflow').setLevel(logging.ERROR)


def get_metrics(data, pred, horizon):
    observed, predicted = utils.get_dfs_commons(data, pred)
    i1, i2, i3, mape = None, None, None, None
    if horizon == 'short':
        i1 = evaluation.mean_abs_error(observed.iloc[:24], predicted.iloc[:24])
        i2 = evaluation.max_abs_error(observed.iloc[:24], predicted.iloc[:24])
        mape = evaluation.mean_abs_percentage_error(observed.iloc[:24], predicted.iloc[:24])

    elif horizon == 'long':
        i1 = evaluation.mean_abs_error(observed.iloc[:24], predicted.iloc[:24])
        i2 = evaluation.max_abs_error(observed.iloc[:24], predicted.iloc[:24])
        i3 = evaluation.mean_abs_error(observed.iloc[24:], predicted.iloc[24:])
        mape = evaluation.mean_abs_percentage_error(observed.iloc[24:], predicted.iloc[24:])

    return i1, i2, i3, mape


def predict_dma(data, dma_name, model_name, model_params, dates_idx, horizon, cols_to_lag, lag_target, cols_to_move_stat,
                window_size, cols_to_decompose, decompose_target, norm_method, clusters_idx):

    models = {'xgb': xgb.XGBRegressor, 'rf': RandomForestRegressor, 'prophet': ProphetForecaster,
              'lstm': LSTMForecaster, 'multi': multi_series.MultiSeriesForecaster}

    def predict(_data, _dma_name, _model_name, _params, _start_train, _start_test, _end_test,
                _cols_to_lag, _cols_to_move_stat, _window_size, _cols_to_decompose, _norm_method, _labels_cluster):
        f = Forecast(data=_data, y_label=_dma_name, cols_to_lag=_cols_to_lag, cols_to_move_stat=_cols_to_move_stat,
                     window_size=_window_size, cols_to_decompose=_cols_to_decompose, norm_method=_norm_method,
                     start_train=_start_train, start_test=_start_test, end_test=_end_test,
                     labels_cluster=_labels_cluster)
        if _model_name == 'lstm':
            return f.format_forecast(f.predict(model=LSTMForecaster, params=_params))
        elif _model_name == 'multi':
            return f.multi_series_predict(params=_params)
        elif _model_name in ['xgb', 'rf', 'prophet']:
            return f.one_step_loop_predict(model=models[_model_name], params=_params)

    t0 = time.time()
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

    labels_cluster = clusters[clusters_idx][dma_name] if model_name == "multi" else []

    predictions = predict(_data=data, _dma_name=dma_name, _model_name=model_name, _params=model_params,
                          _start_train=start_train, _start_test=start_test, _end_test=end_test,
                          _cols_to_lag={**cols_to_lag, **{dma_name: lag_target}}, _cols_to_move_stat=cols_to_move_stat,
                          _window_size=window_size, _cols_to_decompose=_cols_to_decompose, _norm_method=norm_method,
                          _labels_cluster=labels_cluster)

    predictions.columns = [dma_name]
    run_time = time.time() - t0
    try:
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
            'clusters_idx': clusters_idx,
            'i1': i1,
            'i2': i2,
            'i3': i3,
            'mape': mape,
            'run_time': round(run_time, 3)
        })
        return result
    except Exception as e:
        logger.debug(f"dma_name: {dma_name}\nmodel_name: {model_name}\nparams: {model_params}\ndates_idx: {dates_idx}\n"
                     f"horizon: {horizon}\ncols_to_lag: {cols_to_lag}\nlag_target: {lag_target}\n"
                     f"cols_to_move_stat: {cols_to_move_stat}\nwindow_size: {window_size}\n"
                     f"cols_to_decompose: {cols_to_decompose}\n decompose_target: {decompose_target}\n"
                     f"norm_method: {norm_method}\nclusters_idx: {clusters_idx}")
        logger.debug(str(e))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do', type=str, required=True)
    parser.add_argument('--search_params', type=int, required=True)
    parser.add_argument('--dma_idx', type=int, required=False)
    parser.add_argument('--model_name', type=str, required=False)
    parser.add_argument('--dates_idx', type=int, required=False)
    parser.add_argument('--horizon', type=str, required=False)
    parser.add_argument('--norm_methods', nargs='+', type=str, required=True)

    parser.add_argument('--target_lags', nargs='+', type=int, required=True)
    parser.add_argument('--weather_lags', nargs='+', type=int, required=False)
    parser.add_argument('--clusters_idx', nargs='+', type=int, required=False)

    parser.add_argument('--move_stats', type=int, required=False)
    parser.add_argument('--decompose_target', type=int, required=False)
    parser.add_argument('--output_dir', type=str, required=False)

    args = parser.parse_args()
    if args.do == 'experiment':
        run_experiment(args)
    elif args.do == 'test_experiment':
        test_experiment(args)
    elif args.do == 'hyperparam_opt':
        run_hyperparam_opt(args)
    elif args.do == 'random_search':
        random_search(args)

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
    filename = time.strftime("%Y%m%d%H%M%S")
    filename += "".join(f"--{key}-{value}" for key, value in args_dict.items()
                        if key in ['dma_idx', 'model_name', 'dates_idx', 'horizon'])
    filename += f'--{slurm_job_id}'
    filename += '.csv'
    return filename


def run_experiment(args):
    loader = Loader()
    p = Preprocess(loader.inflow, loader.weather, cyclic_time_features=True, n_neighbors=3)
    data = p.data

    results = pd.DataFrame()
    output_dir = utils.validate_dir_path(args.output_dir)
    output_file = generate_filename(args)

    if args.horizon == 'short':
        window_size = 24
    elif args.horizon == 'long':
        window_size = 168
    else:
        window_size = 0

    if args.move_stats:
        include_moving_stats_cols = [True, False]
    else:
        include_moving_stats_cols = [False]

    if args.decompose_target:
        _decompose_target = [True, False]
    else:
        _decompose_target = [False]

    if args.search_params == 1:
        params = grids[args.model_name]['params']
        parameter_sets = list(generate_parameter_sets(params))
    else:
        dma = constants.DMA_NAMES[args.dma_idx]
        parameter_sets = [utils.read_json("multi_series_params.json")[dma[:5]][args.horizon]['params']]

    for params_cfg in parameter_sets:
        for norm in args.norm_methods:
            for wl in args.weather_lags:
                for tl in args.target_lags:
                    for ms in include_moving_stats_cols:
                        for dt in _decompose_target:
                            for c_idx in args.clusters_idx:
                                cols_to_lag = {'Rainfall depth (mm)': wl,
                                               'Air temperature (°C)': wl,
                                               'Windspeed (km/h)': wl,
                                               'Air humidity (%)': wl,
                                               }

                                cols_to_move_stats = constants.WEATHER_COLUMNS if ms else []
                                try:
                                    res = predict_dma(data=data,
                                                      dma_name=constants.DMA_NAMES[args.dma_idx],
                                                      model_name=args.model_name,
                                                      model_params=params_cfg,
                                                      dates_idx=args.dates_idx,
                                                      horizon=args.horizon,
                                                      cols_to_lag=cols_to_lag,
                                                      lag_target=tl,
                                                      cols_to_move_stat=cols_to_move_stats,
                                                      window_size=window_size,
                                                      cols_to_decompose=[],
                                                      decompose_target=dt,
                                                      norm_method=norm,
                                                      clusters_idx=c_idx
                                                      )

                                    results = pd.concat([results, res])
                                    results.to_csv(os.path.join(output_dir, output_file))
                                    del res  # Free up memory
                                    gc.collect()  # Collect garbage after each combination of parameters
                                except Exception as e:
                                    logger.debug(
                                        f"args: {vars(args)}\nparams: {params_cfg}\ncols_to_lag: {cols_to_lag}"
                                        f"\nlag_target: {tl}\ncols_to_move_stat: {cols_to_move_stats}\n"
                                        f"window_size: {window_size}\nnorm_method: {norm}\nclusters_idx: {clusters_idx}")
                                    logger.debug(str(e))
                                    logging.error("Exception occurred", exc_info=True)

                                    # Alternatively, you can format the traceback yourself
                                    tb_info = traceback.format_exc()
                                    logging.error(f"Traceback info: {tb_info}")


def test_experiment(args, n_tests=20):
    loader = Loader()
    p = Preprocess(loader.inflow, loader.weather, cyclic_time_features=True, n_neighbors=3)
    data = p.data

    results = pd.DataFrame()
    output_dir = utils.validate_dir_path(args.output_dir)
    output_file = generate_filename(args)

    params = grids[args.model_name]['params']
    for i, params_cfg in enumerate(generate_parameter_sets(params)):
        res = predict_dma(data=data,
                          dma_name=constants.DMA_NAMES[args.dma_idx],
                          model_name=args.model_name,
                          model_params=params_cfg,
                          dates_idx=args.dates_idx,
                          horizon=args.horizon,
                          cols_to_lag={},
                          lag_target=12,
                          cols_to_move_stat=[],
                          window_size=0,
                          cols_to_decompose=[],
                          decompose_target=False,
                          norm_method='standard',
                          clusters_idx=0
                          )

        results = pd.concat([results, res])
        results.to_csv(os.path.join(output_dir, output_file))
        if i == n_tests:
            break


def run_hyperparam_opt(args):
    if args.model_name == 'multi_series':
        multi_series.tune_dma(constants.DMA_NAMES[args.dma_idx])

    elif args.model_name in ['rf', 'xgb', 'prophet', 'lstm']:
        hyperparam_search.tune_dma(dma=constants.DMA_NAMES[args.dma_idx],
                                   model_name=args.model_name,
                                   dates=constants.EXPERIMENTS_DATES[args.dates_idx],
                                   cols_to_lag={'Air humidity (%)': 6, 'Rainfall depth (mm)': 6,
                                                'Air temperature (°C)': 6, 'Windspeed (km/h)': 6},
                                   lag_target=args.target_lags,
                                   norm_method=args.norm_method,
                                   n_split=3
                                   )


def random_search(args, n=5000):
    loader = Loader()
    p = Preprocess(loader.inflow, loader.weather, cyclic_time_features=True, n_neighbors=3)
    data = p.data

    results = pd.DataFrame()
    output_dir = utils.validate_dir_path(args.output_dir)
    output_file = generate_filename(args)

    if args.horizon == 'short':
        window_size = 24
    elif args.horizon == 'long':
        window_size = 168
    else:
        window_size = 0

    if args.search_params == 1:
        params = grids[args.model_name]['params']
        parameter_sets = list(generate_parameter_sets(params))
    else:
        dma = constants.DMA_NAMES[args.dma_idx]
        parameter_sets = [utils.read_json("multi_series_params.json")[dma[:5]][args.horizon]['params']]

    if not args.model_name == "multi":
        clusters_set = [0]  # arbitrary select one set of clusters, will not be used
    else:
        clusters_set = list(clusters.keys())

    for i in range(n):
        params_cfg = parameter_sets[np.random.randint(low=0, high=len(parameter_sets))]
        norm = args.norm_methods[np.random.randint(low=0, high=len(args.norm_methods))]
        wl = args.weather_lags[np.random.randint(low=0, high=len(args.weather_lags))]
        tl = args.target_lags[np.random.randint(low=0, high=len(args.target_lags))]
        dt = bool(random.getrandbits(1))
        clusters_idx = clusters_set[np.random.randint(low=0, high=len(clusters_set))]

        cols_to_lag = {'Rainfall depth (mm)': wl,
                       'Air temperature (°C)': wl,
                       'Windspeed (km/h)': wl,
                       'Air humidity (%)': wl,
                       }

        try:
            res = predict_dma(data=data,
                              dma_name=constants.DMA_NAMES[args.dma_idx],
                              model_name=args.model_name,
                              model_params=params_cfg,
                              dates_idx=args.dates_idx,
                              horizon=args.horizon,
                              cols_to_lag=cols_to_lag,
                              lag_target=tl,
                              cols_to_move_stat=[],
                              window_size=window_size,
                              cols_to_decompose=[],
                              decompose_target=dt,
                              norm_method=norm,
                              clusters_idx=clusters_idx
                              )

            results = pd.concat([results, res])
            results.to_csv(os.path.join(output_dir, output_file))
        except Exception as e:
            logger.debug(
                f"args: {vars(args)}\nparams: {params_cfg}\ncols_to_lag: {cols_to_lag}"
                f"\nlag_target: {tl}\ncols_to_move_stat: {[]}\n"
                f"window_size: {window_size}\nnorm_method: {norm}\nclusters_idx: {clusters_idx}")
            logger.debug(str(e))
            logging.error("Exception occurred", exc_info=True)

            # Alternatively, you can format the traceback yourself
            tb_info = traceback.format_exc()
            logging.error(f"Traceback info: {tb_info}")


if __name__ == "__main__":
    global_seed = 42
    os.environ['PYTHONHASHSEED'] = str(global_seed)
    random.seed(global_seed)
    np.random.seed(global_seed)
    parse_args()