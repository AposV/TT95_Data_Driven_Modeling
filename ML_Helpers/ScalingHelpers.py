from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np


class MinMaxScaler3D(MinMaxScaler):

    def transform3D(self, X):
        x = np.reshape(X, newshape=(X.shape[0]*X.shape[1], X.shape[2]))
        return np.reshape(super().transform(x), newshape=X.shape)


class StandardScaler3D(StandardScaler):

    def transform3D(self, X, copy=None):
        x = np.reshape(X, newshape=(X.shape[0] * X.shape[1], X.shape[2]))
        return np.reshape(super().transform(x, copy=copy), newshape=X.shape)