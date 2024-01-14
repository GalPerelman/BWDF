import ast
import glob
import math
import os
import json

import numpy as np
import pytz
import pandas as pd
import datetime
from typing import Sequence
from ast import literal_eval

import constants


def validate_dir_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def read_json(file):
    with open(file, encoding='utf-8') as f:
        return json.load(f)


def import_preprocessed(path):
    data = pd.read_csv(path, index_col=0)
    data.index = pd.to_datetime(data.index, utc=True)
    data.index = data.index.tz_convert(constants.TZ)
    return data


def localize_datetime(d):
    """
    This function get a datetime.datetime object and do the following:
    - convert it to time zone aware
    - convert it to be dst aware

    return: datetime.datetime
    """
    tz_object = pytz.timezone(str(constants.TZ))
    aware_dt = tz_object.localize(d)
    return aware_dt


def get_test_dates(test_name: str):
    """

    test_name: Must be one of: 'w1', 'w2', 'w3', 'w4'
    return: tuple of 3 datetime.datetime (start_test, end_short_test, end_long_test)
    """
    start_test = constants.TEST_TIMES[test_name]['start']
    end_short_test = start_test + datetime.timedelta(hours=24)
    end_long_test = constants.TEST_TIMES[test_name]['end']
    return start_test, end_short_test, end_long_test


def drop_other_dmas(data, y_label):
    cols_to_drop = list(set(constants.DMA_NAMES) - set([y_label]))
    data = data.drop(cols_to_drop, axis=1)
    return data


def num_hours_between_timestamps(t1, t2):
    """
    Calculate the number of hours between two timestamps where t2 > t1
    param t1:       datetime.datetime, period start
    param t2:       datetime.datetime, period end
    return:         int, number of hours
    """
    diff = t2 - t1
    days, seconds = diff.days, diff.seconds
    hours = int(days * 24 + seconds // 3600)
    return hours


def record_results(dma: str, short_model_name: str, long_model_name: str, dates: dict, lags: dict, norm_method: str,
                   pred_type: str, cols_to_move_stat: list, cols_to_decompose: list, clusters_idx: int,
                   score: Sequence):

    # WINDOW_SIZE IS ACCORDING TO PREDICTION HORIZON
    result = pd.DataFrame({
        'dma': dma,
        'short_model_name': short_model_name,
        'long_model_name': long_model_name,
        'start_train': dates['start_train'],
        'start_test': dates['start_test'],
        'end_test': dates['end_test'],
        'lags': [lags],
        'norm': norm_method,
        'pred_type': pred_type,
        'cols_to_move_stat': [cols_to_move_stat],
        'cols_to_decompose': [cols_to_decompose],
        'clusters_idx': clusters_idx,
        'i1': score[0],
        'i2': score[1],
        'i3': score[2],
    }, index=[datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")])

    df = pd.read_csv("models_scores.csv", index_col=0)
    df = pd.concat([df, result])
    df.to_csv("models_scores.csv", index=True)


def get_dfs_commons(df1, df2):
    common_cols = df1.columns.intersection(df2.columns).to_list()
    common_rows = pd.merge(df1, df2, left_index=True, right_index=True, how='right').index
    df1 = df1.loc[common_rows, common_cols]
    df2 = df2.loc[common_rows, common_cols]
    return df1, df2


def decompose_dictionaries_column(df, column_name, prefix='_'):
    df.reset_index(inplace=True)
    try:
        # convert str to dictionary
        df[column_name] = df[column_name].apply(ast.literal_eval)
    except (SyntaxError, ValueError):
        pass

    decomposed = pd.json_normalize(df[column_name])
    decomposed.columns = [prefix + col for col in decomposed.columns]
    df = pd.concat([df.drop([column_name], axis=1), decomposed], axis=1)
    return df


def collect_experiments(dir_path, p, dmas, horizon, dates_idx, models, abs_n=None):
    df = pd.DataFrame()

    def get_param(raw_str):
        return raw_str.split('-')[1]

    for i, fname in enumerate(glob.glob(dir_path + "/*.csv")):
        _prefix, _dma_idx, _model_name, _dates_idx, _horizon, _slurm_id = fname.split('--')
        _dma_idx = int(get_param(_dma_idx))
        _model_name = get_param(_model_name)
        _dates_idx = int(get_param(_dates_idx))
        _horizon = get_param(_horizon)
        if (_dma_idx in dmas) and (_model_name in models) and (_dates_idx in dates_idx) and (_horizon == horizon):
            temp = pd.read_csv(fname, index_col=0, engine="python", on_bad_lines='skip')
            temp = temp.sort_values('i1')
            # temp = temp.head(max(50, int(len(temp) * (p / 100))))
            temp = decompose_dictionaries_column(temp, column_name='model_params', prefix='param_')
            temp = decompose_dictionaries_column(temp, column_name='lags', prefix='lags_')
            df = pd.concat([df, temp])

    def select_smallest_n_percent(group, n_percent, target_col):
        if abs_n is None:
            n = max(50, math.ceil(len(group) * n_percent / 100))
        else:
            n = abs_n
        return group.nsmallest(n, target_col)

    df = df.drop_duplicates()
    target_col = 'i1' if horizon == 'short' else 'i3'
    df = df.groupby(['dma', 'model_name', 'dates_idx'], group_keys=False).apply(
        lambda x: select_smallest_n_percent(x, n_percent=p, target_col=target_col))
    return df


def experiment_to_json(csv_path, horizon, models, export_path):
    """
    Nested parsing of experiments csv files to construct final candidates
    The csv files are generated with the select_models function in graphs where the number of models in each category
    (dma, model_name, dates) need to be defined
    """
    df = pd.read_csv(csv_path, index_col=0)
    df["cols_to_move_stat"] = df["cols_to_move_stat"].apply(lambda x: [] if pd.isna(x) else literal_eval(x))
    df["cols_to_decompose"] = df["cols_to_decompose"].apply(lambda x: [] if pd.isna(x) else literal_eval(x))

    for col in df.columns:
        if col.startswith("lags_"):
            df[col] = df[col].fillna(0)

    candidates = {}

    for dma in constants.DMA_NAMES:
        temp = df.loc[df['dma'] == dma]

        dma_candidates = []
        for i, row in temp.iterrows():
            if row["model_name"] not in models:
                continue
            params = {col[6:]: row[col] for col in temp if col.startswith("param_") and not pd.isnull(row[col])}
            params = {name: int(value) if isinstance(value, (int, float)) and np.floor(value) == value else value for
                      name, value in params.items()}

            dma_candidates.append(
                {
                    "model_name": row["model_name"],
                    "params": params,
                    "cols_to_move_stat": row["cols_to_move_stat"],
                    "cols_to_decompose": row["cols_to_decompose"],
                    "decompose_target": True if dma in row["cols_to_decompose"] else False,
                    "norm_method": row["norm"],
                    "lags": {"Rainfall depth (mm)": int(row["lags_Rainfall depth (mm)"]),
                             "Air temperature (°C)": int(row["lags_Air temperature (°C)"]),
                             "Windspeed (km/h)": int(row["lags_Windspeed (km/h)"]),
                             "Air humidity (%)": int(row["lags_Air humidity (%)"])
                             },
                    "lag_target": int(row[f"lags_{dma}"]),
                    "clusters_idx": int(row["clusters_idx"]) if not pd.isnull(row["clusters_idx"]) else None
                }
            )
        candidates[dma] = {horizon: dma_candidates}

    with open(export_path, 'w', encoding='utf8') as file:
        json.dump(candidates, file, indent=4, ensure_ascii=False)
