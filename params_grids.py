from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn_genetic.space import Continuous, Categorical, Integer

from prophet_model import ProphetForecaster
from lstm_model import LSTMForecaster

xgb.set_config(verbosity=0)

grids = {
    'rf':
        {
            'model': RandomForestRegressor(),

            'params':
                {'bootstrap': [True, False],
                 'max_depth': [10, 20, 30, 50, None],
                 'min_samples_leaf': [1, 2, 4],
                 'min_samples_split': [2, 5, 10],
                 'n_estimators': [25, 50, 100, 200, 500],
                 'max_features': [10, 'sqrt', 'log2']

                 },

            'ga_params':
                {'bootstrap': Categorical([True, False]),
                 'max_depth': Integer(10, 100),
                 'min_samples_leaf': Integer(1, 4),
                 'min_samples_split': Integer(2, 10),
                 'n_estimators': Integer(20, 50)
                 },

        },

    'xgb':
        {
            'model': xgb.XGBRegressor(verbosity=0, silent=True),
            'params':
                {
                    'bootstrap': [True, False],
                    "max_depth": [3, 5, 10, 15],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "n_estimators": [50, 100, 200, 300, 500],
                    "reg_alpha": [0, 0.01, 0.1, 0.5, 5],
                    "reg_lambda": [0, 0.01, 0.1, 0.5, 5],
                    # "min_sample_leaf": [2, 3, None],
                    # "min_sample_split": [2, 3, None],
                    "min_child_weight": [1, 3, 5],
                },

            'ga_params':
                {
                    'bootstrap': Categorical([True, False]),
                    'max_depth': Integer(10, 20),
                    'learning_rate': Continuous(0.01, 0.4),
                    'n_estimators': Integer(20, 50)
                },
        },

    'prophet':
        {
            'model': ProphetForecaster(),
            'params':
                {
                    'seasonality_mode': ['additive', 'multiplicative'],
                    'daily_seasonality': [True, False],
                    'weekly_seasonality': [True, False],
                    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0, 20.0],
                    'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
                    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
                }
        },

    'lstm':
        {
            'model': LSTMForecaster(),
            'params':
                {
                    "look_back": [12, 24, 48],
                    "epochs": [10],
                    "batch_size": [24, 48],
                    "units": [50, 100, 150],
                    "dropout": [0.1, 0.2, 0.3]
                }
        }
}
