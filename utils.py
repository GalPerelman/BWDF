import os
import json
import pytz
import pandas as pd
import datetime

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