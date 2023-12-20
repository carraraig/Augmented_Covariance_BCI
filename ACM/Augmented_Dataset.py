from pyriemann.spatialfilters import Xdawn
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class AugmentedDataset(BaseEstimator, TransformerMixin):

    def __init__(self, order=1, lag=1):

        self.order = order
        self.lag = lag

    def fit(self, X, y):

        return self

    def transform(self, X):


        if self.order == 1:
            X_fin = X
        else:
            X_fin = []

            for i in np.arange(X.shape[0]):
                X_p = X[i][:, : -self.order*self.lag]
                for p in np.arange(1, self.order):
                    X_p = np.append(X_p, X[i][:, p*self.lag: -(self.order - p)*self.lag], axis=0)
                X_fin.append(X_p)
            X_fin = np.array(X_fin)

        return X_fin


class AugmentedDataset2(BaseEstimator, TransformerMixin):

    def __init__(self, order=1, lag=1):

        self.order = order
        self.lag = lag

    def fit(self, X, y):

        return self

    def transform(self, X):


        if self.order == 1:
            X_fin = X
        else:
            X_p = X[:, :, : -self.order * self.lag]
            X_p = np.concatenate([X_p] + [X[:, :, p * self.lag: -(self.order - p) * self.lag] for p in range(1, self.order)], axis=1)
            X_fin = X_p

        return X_fin

