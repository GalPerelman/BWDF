import timeit
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Categorical, Integer
from sklearn_genetic.callbacks import ProgressBar
import xgboost as xgb
import warnings
warnings.filterwarnings(action="ignore")
xgb.set_config(verbosity=0)


class Searcher:
    def __init__(self, model_name, model_info, x_train, y_train, y_label, name, n_splits=3):
        self.model_name = model_name
        self.model = model_info['model']
        self.params = model_info['params']
        self.ga_params = model_info['ga_params']
        self.x_train = x_train
        self.y_train = y_train
        self.y_label = y_label
        self.name = name
        self.n_splits = n_splits

        self.tscv = TimeSeriesSplit(n_splits=self.n_splits)
        self.results = pd.DataFrame()

    def grid_search(self):
        gs_model = GridSearchCV(estimator=self.model, param_grid=self.params, cv=self.tscv, verbose=3,
                                scoring='neg_mean_absolute_error', n_jobs=-1)

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
