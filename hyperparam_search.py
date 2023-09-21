import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Categorical, Integer
from sklearn_genetic.callbacks import ProgressBar
import xgboost as xgb
import warnings
warnings.filterwarnings(action="ignore")
xgb.set_config(verbosity=0)


class Searcher:
    def __init__(self, model_name, model_info, x_train, y_train, n_splits=3):
        self.model_name = model_name
        self.model = model_info['model']
        self.params = model_info['params']
        self.ga_params = model_info['ga_params']
        self.x_train = x_train
        self.y_train = y_train
        self.n_splits = n_splits

        self.tscv = TimeSeriesSplit(n_splits=self.n_splits)
        self.results = pd.DataFrame()

    def grid_search(self):
        gs_model = GridSearchCV(estimator=self.model, param_grid=self.params, cv=self.tscv, verbose=3,
                                scoring='neg_mean_absolute_error')

        gs_model.fit(self.x_train, self.y_train)
        print(gs_model.best_params_, gs_model.best_score_)
        self.record(self.model_name, gs_model.best_params_, "neg_mean_absolute_error", gs_model.best_score_)

    def ga_search(self):
        ga_model = GASearchCV(estimator=self.model, param_grid=self.ga_params, scoring="neg_mean_absolute_error",
                              cv=self.tscv, verbose=True, population_size=10, generations=25, n_jobs=-1)

        ga_model.fit(self.x_train, self.y_train)
        print(self.model_name, ga_model.best_params_, ga_model.best_score_)
        self.record(self.model_name, ga_model.best_params_, "neg_mean_absolute_error", ga_model.best_score_)

    def record(self, model_name, best_params, score_metric, score_value):
        df = pd.DataFrame({'model_name': model_name,
                           'best_params': [best_params],
                           'score_metric': score_metric,
                           'score_value': score_value},
                          index=[len(self.results)])
        self.results = pd.concat([self.results, df])

    def export_results(self):
        self.results.to_csv('hp.csv')
