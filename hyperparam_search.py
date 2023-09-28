import timeit
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn_genetic import GASearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn_genetic.space import Continuous, Categorical, Integer
from sklearn_genetic.callbacks import ProgressBar


import evaluation
import models
from preprocess import Preprocess


class Searcher:
    def __init__(self,
                 model_name,
                 model_info,
                 data,
                 y_label,
                 n_lags,
                 start_train,
                 start_test,
                 end_test,
                 name,
                 n_splits=3):

        self.model_name = model_name
        self.model = model_info['model']
        self.params = model_info['params']
        self.ga_params = model_info['ga_params']
        self.data = data
        self.y_label = y_label
        self.n_lags = n_lags
        self.start_train = start_train
        self.start_test = start_test
        self.end_test = end_test
        self.name = name
        self.n_splits = n_splits

        self.scoring = {
            'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
            'MAX_E': make_scorer(evaluation.max_abs_error, greater_is_better=False)
        }

        self.x_train, self.y_train, self.x_test, self.y_test = Preprocess.by_label(data=self.data,
                                                                                   y_label=self.y_label,
                                                                                   n_lags=self.n_lags,
                                                                                   start_train=self.start_train,
                                                                                   start_test=self.start_test,
                                                                                   end_test=self.end_test)

        self.tscv = TimeSeriesSplit(n_splits=self.n_splits)
        self.results = pd.DataFrame()

    def grid_search(self):
        gs_model = GridSearchCV(estimator=self.model, param_grid=self.params, cv=self.tscv, verbose=3,
                                scoring=self.scoring, refit='MAE', n_jobs=-1)

        gs_model.fit(self.x_train, self.y_train)
        print(gs_model.best_params_, gs_model.best_score_)
        results = pd.DataFrame(gs_model.cv_results_)
        results.to_csv(f'{self.name}_{self.y_label[:-6]}_{self.model_name}.csv', index=False)

    def ga_search(self):
        ga_model = GASearchCV(estimator=self.model, param_grid=self.ga_params, scoring="neg_mean_absolute_error",
                              cv=self.tscv, verbose=True, population_size=10, generations=25, n_jobs=-1)

        ga_model.fit(self.x_train, self.y_train)
        print(self.model_name, ga_model.best_params_, ga_model.best_score_)
        results = pd.DataFrame(ga_model.cv_results_)
        results.to_csv(f'{self.y_label[:-6]}_{self.model_name}.csv', index=False)

    def record(self, model_name, best_params, score_metric, score_value):
        df = pd.DataFrame({'model_name': model_name,
                           'best_params': [best_params],
                           'score_metric': score_metric,
                           'score_value': score_value},
                          index=[len(self.results)])
        self.results = pd.concat([self.results, df])

    def export_results(self, export_file_name):
        self.results.to_csv(f'{export_file_name}_params.csv')

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


def tune_dma(data, dma_name, n_lags, start_train, start_test, end_test):
    for model_name, model_info in models.items():
        searcher = Searcher(model_name=model_name,
                            model_info=model_info,
                            data=data,
                            y_label=dma_name,
                            n_lags=n_lags,
                            start_train=start_train,
                            start_test=start_test,
                            end_test=end_test,
                            )

        searcher.grid_search()

