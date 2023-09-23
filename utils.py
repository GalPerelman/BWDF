import os


def validate_dir_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path