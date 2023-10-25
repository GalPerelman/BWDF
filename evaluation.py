import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error as mae

import constants
import utils


def mean_abs_error(observed, predicted):
    return mae(observed, predicted)


def max_abs_error(observed, predicted):
    return np.max(np.abs(observed - predicted))


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
    one_week_score(observed=data, predicted=pred)