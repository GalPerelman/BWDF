import os
import json
import pandas as pd

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