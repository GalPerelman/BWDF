import pandas as pd
from sklearn.impute import KNNImputer


class Preprocess:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def data_completion(self, data: pd.DataFrame, n_neighbors: int):
        knn_impute = KNNImputer(n_neighbors=n_neighbors)
        data_imputed = knn_impute.fit_transform(data)
        data = pd.DataFrame(data_imputed, columns=data.columns, index=self.data.index)
        return data


