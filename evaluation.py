from sklearn.metrics import mean_absolute_error as mae


def mean_abs_error(observed, predicted):
    return mae(observed, predicted)


def max_abs_error(observed, predicted):
    diff = observed - predicted
    return max(max(diff), -min(diff))