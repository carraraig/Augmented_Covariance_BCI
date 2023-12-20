from pyriemann.spatialfilters import Xdawn
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.preprocessing import StandardScaler


class StandardScaler_Epoch(BaseEstimator, TransformerMixin):

    def __init__(self):
        """Init."""

    def fit(self, X, y):

        return self

    def transform(self, X):

        X_fin = []

        for i in np.arange(X.shape[0]):
            X_p = StandardScaler().fit_transform(X[i])
            X_fin.append(X_p)
        X_fin = np.array(X_fin)

        return X_fin
