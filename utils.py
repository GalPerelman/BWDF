import os
import json


def validate_dir_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def read_json(file):
    with open(file) as f:
        return json.load(f)