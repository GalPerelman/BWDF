import datetime
import os

import pandas as pd

import constants
import evaluation
import forecast
import utils
from clusters import clusters
from data_loader import Loader
from preprocess import Preprocess


class CV:
    def __init__(self, inflow_data_file, weather_data_file, outliers_config,
                 candidates_path, folding_start_date, repeats, hours_step_size, files_suffix, output_dir):
        self.inflow_data_file = inflow_data_file
        self.weather_data_file = weather_data_file
        self.outliers_config = outliers_config
        self.candidates_path = candidates_path
        self.folding_start_date = folding_start_date
        self.repeats = repeats
        self.hours_step_size = hours_step_size
        self.files_suffix = files_suffix
        self.output_dir = utils.validate_dir_path(output_dir)

        self.candidates = utils.read_json(self.candidates_path)
        self.raw_data = self.load_data()

        self.window_size = {'short': 24, 'long': 168}

    def load_data(self):
        loader = Loader(inflow_data_file=self.inflow_data_file, weather_data_file=self.weather_data_file)
        p = Preprocess(loader.inflow, loader.weather, cyclic_time_features=True, n_neighbors=3,
                       outliers_config=self.outliers_config)

        return p.data

    def folding_pred(self, dma, horizon, model_config, model_idx):
        fold_results = pd.DataFrame()
        cols_to_lag = {**model_config["lags"], **{dma: model_config["lag_target"]}}
        if model_config["decompose_target"]:
            _cols_to_decompose = model_config["cols_to_decompose"] + [dma]
        else:
            _cols_to_decompose = model_config["cols_to_decompose"]

        if model_config["model_name"] == "multi":
            label_clusters = clusters[model_config["clusters_idx"]][dma]
        else:
            label_clusters = None

        for i in range(self.repeats):
            start_test = self.folding_start_date + i * datetime.timedelta(hours=self.hours_step_size)
            pred = forecast.predict_dma(data=self.raw_data,
                                        dma_name=dma,
                                        model_name=model_config["model_name"],
                                        params=model_config['params'],
                                        start_train=self.raw_data.index.min(),
                                        start_test=start_test,
                                        end_test=start_test + datetime.timedelta(hours=self.window_size[horizon]),
                                        cols_to_lag=cols_to_lag,
                                        cols_to_move_stat=model_config["cols_to_move_stat"],
                                        window_size=self.window_size[horizon],
                                        cols_to_decompose=_cols_to_decompose,
                                        norm_method=model_config["norm_method"],
                                        labels_cluster=label_clusters,
                                        pred_type="step-ahead"
                                        )

            # manually adjustments - DMA A
            if (dma == constants.DMA_NAMES[0] and horizon == 'short'
                    and self.outliers_config["manual_adjustments"][dma]['short']):
                pred.iloc[0] = 0.0505 * pred.sum() + 4.85

            pred.columns = [dma]

            i1, i2, i3, mape = evaluation.get_metrics(self.raw_data, pred, horizon=horizon)
            df = pd.DataFrame({
                'model_idx': model_idx,
                'dma': dma,
                'model_name': model_config["model_name"],
                'start_train': self.raw_data.index.min(),
                'start_test': start_test,
                'end_test': start_test + datetime.timedelta(hours=self.window_size[horizon]),
                'params': [model_config['params']],
                'lags': [model_config["lags"]],
                'lag_target': model_config["lag_target"],
                'norm': model_config["norm_method"],
                'cols_to_move_stat': [model_config["cols_to_move_stat"]],
                'cols_to_decompose': [model_config["cols_to_decompose"]],
                'decompose_target': model_config["decompose_target"],
                'clusters_idx': model_config["clusters_idx"],
                'i1': i1,
                'i2': i2,
                'i3': i3,
                'mape': mape
            }, index=[len(fold_results)])

            fold_results = pd.concat([fold_results, df])
            print(fold_results)

        return fold_results

    def run_single_experiment(self, dma, horizon, reverse=False):
        all_result = pd.DataFrame()

        if reverse:
            models_configs = self.candidates[dma][horizon][::-1]
        else:
            models_configs = self.candidates[dma][horizon]

        for model in models_configs:
            fold_results = self.folding_pred(dma, horizon, model, model["model_idx"])
            all_result = pd.concat([all_result, fold_results])
            all_result.to_csv(os.path.join(self.output_dir, f"cv_dma-{dma[:5]}_{horizon}_{self.files_suffix}.csv"))

    def run_all(self, horizon):
        for dma in constants.DMA_NAMES:
            self.run_single_experiment(dma, horizon)


if __name__ == "__main__":
    df = utils.collect_experiments("experiment_output/v3", p=0, dmas=[_ for _ in range(10)], horizon='short',
                                   dates_idx=[0, 2, 3, 4], models=['multi', 'xgb', 'lstm', 'prophet'], abs_n=3)
    df.to_csv("experiments_analysis/candidates_short_v1.csv", encoding='utf8')

    df = utils.collect_experiments("experiment_output/v3", p=0, dmas=[_ for _ in range(10)], horizon='long',
                                   dates_idx=[0, 2, 3, 4], models=['multi', 'xgb', 'lstm', 'prophet'], abs_n=3)
    df.to_csv("experiments_analysis/candidates_long_v1.csv", encoding='utf8')

    utils.experiment_to_json(csv_path="experiments_analysis/candidates_short_v1.csv", horizon="short",
                             models=['lstm', 'xgb', 'multi', 'prophet'],
                             export_path="experiments_analysis/candidates_short_v1.json",
                             # constant_lags={"Air temperature (°C)": 6, "Air humidity (%)": 6}
                             )

    utils.experiment_to_json(csv_path="experiments_analysis/candidates_long_v1.csv", horizon="long",
                             models=['lstm', 'xgb', 'multi', 'prophet'],
                             export_path="experiments_analysis/candidates_long_v1.json",
                             # constant_lags={"Air temperature (°C)": 6, "Air humidity (%)": 6}
                             )

    # start = constants.TZ.localize(datetime.datetime(2022, 7, 4, 0, 0))
    # cv = CV(candidates_path="experiments_analysis/candidates_short_constant_lags.json",
    #         folding_start_date=start, repeats=14, hours_step_size=24, files_suffix="constant_lags")
    # cv.run_all(horizon='short')
    
    # cv = CV(candidates_path="experiments_analysis/candidates_long.json",
    #         folding_start_date=start, repeats=14, hours_step_size=24)
    # cv.run_all(horizon='long')

