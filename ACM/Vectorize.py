from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class Vectorize(BaseEstimator, TransformerMixin):

    def __init__(self):
        """Init."""

    def fit(self, X, y):

        return self

    def transform(self, X):

        X_fin = []

        for i in np.arange(X.shape[0]):
            X_fin.append(X[i].flatten())
        X_fin = np.array(X_fin)
        # Need to reshape in a suitable way n_epoch x feature_in_each_epoch
        # X_fin = X_fin.reshape(len(X_fin), -1)

        return X_fin