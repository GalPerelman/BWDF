import argparse
import os
import random
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import forecast
from forecast import *
from data_loader import Loader

warnings.filterwarnings("ignore")


def validate_input(args):
    if args.predict == 'test':
        args.dates = constants.TEST_TIMES[args.test_name]
        args.export_path = f"forecast-{args.predict}-{args.test_name}"
    elif args.predict == 'experiment':
        args.dates = constants.EXPERIMENTS_DATES[args.experiment_idx]
        args.export_path = f"forecast-{args.predict}-{args.experiment_idx}"
    else:
        raise Exception('--predict must be one of: "experiment", "test"')

    return args


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inflow_data_file', type=str, required=True)
    parser.add_argument('--weather_data_file', type=str, required=True)
    parser.add_argument('--predict', type=str, required=True, choices=['experiment', 'test'])
    parser.add_argument('--models_config', type=str, required=True)
    parser.add_argument('--test_name', type=str, required=False, choices=['w1', 'w2', 'w3', 'w4'])
    parser.add_argument('--experiment_idx', type=int, required=False)
    parser.add_argument('--plot', type=bool, required=False, default=True)
    parser.add_argument('--export', type=bool, required=False, default=True)
    args = parser.parse_args()

    args = validate_input(args)
    models_config = utils.read_json(args.models_config)

    loader = Loader(inflow_data_file=args.inflow_data_file, weather_data_file=args.weather_data_file)
    prep = Preprocess(loader.inflow, loader.weather, cyclic_time_features=True, n_neighbors=3,
                      outliers_config=models_config)

    data = prep.data
    forecast.predict_all_dmas(data=data, dates=args.dates, models=models_config, plot=args.plot, export=args.export,
                              export_path=args.export_path)


if __name__ == "__main__":
    global_seed = 42
    os.environ['PYTHONHASHSEED'] = str(global_seed)
    random.seed(global_seed)
    np.random.seed(global_seed)
    run()
