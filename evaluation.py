import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape

import constants
import utils


def mean_abs_error(observed, predicted):
    e = mae(observed, predicted, multioutput='raw_values')
    if observed.shape[1] == 1:
        return e.item()
    else:
        return e


def max_abs_error(observed, predicted):
    e = np.max(np.abs(observed - predicted), axis=0).values
    if observed.shape[1] == 1:
        return e.item()
    else:
        return e


def mean_abs_percentage_error(observed, predicted):
    e = mape(observed, predicted, multioutput='raw_values')
    if observed.shape[1] == 1:
        return e.item()
    else:
        return e


def one_week_score(observed: pd.DataFrame, predicted: pd.DataFrame):
    # validate data
    if len(predicted) != 168:
        raise "Error: data is not of one week"

    common_cols = observed.columns.intersection(predicted.columns).to_list()
    common_rows = pd.merge(observed, predicted, left_index=True, right_index=True, how='right').index
    observed = observed.loc[common_rows, common_cols]
    predicted = predicted.loc[common_rows, common_cols]

    i1 = mean_abs_error(observed.iloc[:24], predicted.iloc[:24])
    i2 = max_abs_error(observed.iloc[:24], predicted.iloc[:24])
    i3 = mean_abs_error(observed.iloc[24:], predicted.iloc[24:])
    return i1, i2, i3


if __name__ == "__main__":
    # usage example
    data = utils.import_preprocessed("resources/preprocessed_data.csv")[constants.DMA_NAMES]
    pred = utils.import_preprocessed("results.csv")
    print(one_week_score(observed=data, predicted=pred))