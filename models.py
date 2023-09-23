from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
import xgboost as xgb

from sklearn_genetic.space import Continuous, Categorical, Integer

# from statsmodels.tsa.statespace.sarimax import SARIMAX
xgb.set_config(verbosity=0)


models = {
    'rf':
        {
            'model': RandomForestRegressor(),

            'params':
                {'bootstrap': [True, False],
                 'max_depth': [10, 20, 30, 50, None],
                 'min_samples_leaf': [1, 2, 4],
                 'min_samples_split': [2, 5, 10],
                 'n_estimators': [25, 50, 100, 200, 500],
                 'max_features': ['auto', 'sqrt', 'log2']

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
                 "max_depth": [4, 5, 6, 7, 8],
                 "learning_rate": [0.01, 0.05, 0.1, 0.2],
                 "n_estimators": [25, 50, 100, 200, 500],
                 "colsample_bytree": [0.4, 0.5, 0.7],
                 "reg_alpha": [0, 0.1, 0.5],
                 "min_sample_leaf": [2, 3, None],
                 "min_sample_split": [2, 3, None]
                 },

            'ga_params':
                {
                 'bootstrap': Categorical([True, False]),
                 'max_depth': Integer(10, 20),
                 'learning_rate': Continuous(0.01, 0.4),
                 'colsample_bytree': Continuous(0.3, 0.7),
                 'n_estimators': Integer(20, 50)
                 },
        }
}
