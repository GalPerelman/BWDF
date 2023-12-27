import os
import json
import pytz
import pandas as pd
import datetime
from typing import Sequence

import constants


def validate_dir_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def read_json(file):
    with open(file) as f:
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
                   pred_type: str, cols_to_move_stat: list, cols_to_decompose: list, score: Sequence):

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
