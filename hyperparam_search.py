import timeit

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn_genetic import GASearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn_genetic.space import Continuous, Categorical, Integer
from sklearn_genetic.callbacks import ProgressBar

import utils
import constants
import evaluation
from data_loader import Loader
from params_grids import grids
from preprocess import Preprocess


class Searcher:
    def __init__(self, model_name, model_info, data, y_label, cols_to_lag, norm_method,
                 start_train, start_test, end_test, n_splits=3):

        self.model_name = model_name
        self.model = model_info['model']
        self.params = model_info['params']
        self.data = data
        self.y_label = y_label
        self.cols_to_lag = cols_to_lag
        self.norm_method = norm_method
        self.start_train = start_train
        self.start_test = start_test
        self.end_test = end_test
        self.n_splits = n_splits

        self.pred_term = self.is_short_or_long()
        self.window_size = 24 if self.pred_term == 'short' else 168

        self.scoring = {
            'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
            'MAX_E': make_scorer(evaluation.max_abs_error, greater_is_better=False)
        }

        temp_data = self.data.copy()
        temp_data = Preprocess.drop_other_dmas(temp_data, self.y_label)
        temp_data, lagged_cols = Preprocess.lag_features(temp_data, cols_to_lag=self.cols_to_lag)
        if self.norm_method:
            norm_cols = constants.WEATHER_COLUMNS + lagged_cols + [self.y_label]
        else:
            norm_cols = None

        preprocessed = Preprocess.split_data(data=temp_data, y_label=self.y_label, start_train=self.start_train,
                                             start_test=self.start_test, end_test=self.end_test,
                                             norm_method=self.norm_method, norm_cols=norm_cols,
                                             norm_param=self.window_size
                                             )

        # unpack preprocessed
        self.x_train, self.y_train, self.x_test, self.y_test, self.scalers = preprocessed

        self.tscv = TimeSeriesSplit(n_splits=self.n_splits)
        self.results = pd.DataFrame()

    def is_short_or_long(self):
        """
        Is model for short term (24 hr) or long term (168 hr)
        return: str 'short' or 'long'
        """
        if self.end_test - self.start_test == datetime.timedelta(hours=24):
            return 'short'
        elif self.end_test - self.start_test == datetime.timedelta(hours=168):
            return 'long'

    def grid_search(self):
        gs_model = GridSearchCV(estimator=self.model, param_grid=self.params, cv=self.tscv, verbose=3,
                                scoring=self.scoring, refit='MAE', n_jobs=-1)

        gs_model.fit(self.x_train, self.y_train)
        results = pd.DataFrame(gs_model.cv_results_)
        results.to_csv(f'{self.pred_term}_{self.y_label[:-6]}_{self.model_name}.csv', index=False)

    def ga_search(self):
        ga_model = GASearchCV(estimator=self.model, param_grid=self.ga_params, scoring="neg_mean_absolute_error",
                              cv=self.tscv, verbose=True, population_size=10, generations=25, n_jobs=-1)

        ga_model.fit(self.x_train, self.y_train)
        results = pd.DataFrame(ga_model.cv_results_)
        results.to_csv(f'{self.y_label[:-6]}_{self.model_name}.csv', index=False)

    def record(self, model_name, best_params, score_metric, score_value):
        df = pd.DataFrame({'model_name': model_name,
                           'best_params': [best_params],
                           'score_metric': score_metric,
                           'score_value': score_value},
                          index=[len(self.results)])
        self.results = pd.concat([self.results, df])

    def estimate_gridsearch_time(self):
        times = []
        for _ in range(5):
            start = timeit.default_timer()
            self.model.fit(self.x_train, self.y_train)
            self.model.score(self.x_train, self.y_train)
            times.append(timeit.default_timer() - start)
            print(times)

        single_train_time = np.array(times).mean()  # seconds
        print(single_train_time)

        combos = 1
        for vals in self.params.values():
            combos *= len(vals)

        num_models = combos * self.n_splits
        seconds = num_models * single_train_time
        minutes = seconds / 60
        hours = minutes / 60
        print(f"Estimated search time {hours:.1f} hours")


def tune_dma(dma, model_name: str, dates: dict, cols_to_lag, lag_target, norm_method, n_split):
    loader = Loader()
    preprocess = Preprocess(loader.inflow, loader.weather, cyclic_time_features=True, n_neighbors=3)
    data = preprocess.data

    model_info = grids[model_name]

    # tune for short term
    searcher = Searcher(model_name=model_name,
                        model_info=model_info,
                        data=data,
                        y_label=dma,
                        cols_to_lag={**cols_to_lag, **{dma: lag_target}},
                        norm_method=norm_method,
                        start_train=dates['start_train'],
                        start_test=dates['start_test'],
                        end_test=dates['start_test'] + datetime.timedelta(hours=24),
                        n_splits=n_split
                        )

    searcher.grid_search()

    # tune for long term
    searcher = Searcher(model_name=model_name,
                        model_info=model_info,
                        data=data,
                        y_label=dma,
                        cols_to_lag={**cols_to_lag, **{dma: lag_target}},
                        norm_method=norm_method,
                        start_train=dates['start_train'],
                        start_test=dates['start_test'],
                        end_test=dates['start_test'] + datetime.timedelta(hours=168),
                        n_splits=n_split
                        )

    searcher.grid_search()
