import numpy as np


class BaseModel(object):

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, x, y):
        return NotImplemented

    def predict(self, x):
        return NotImplemented


class ConstantModel(BaseModel):

    def fit(self, x, y):
        self.coef_ = y.mean()

    def predict(self, x):
        return np.ones(len(x)) * self.coef_


class LinearRegressionNumpy(BaseModel):

    def fit(self, x, y):
        if len(x.shape) == 1:
            self.coef_ = np.dot(x.T, y) / np.dot(x.T, x)
        else:
            self.coef_ = np.linalg.inv(np.dot(x.T, x)).dot(np.dot(x.T, y))

    def predict(self, x):
        if len(x.shape) == 1:
            return self.coef_ * x
        else:
            return self.coef_.dot(x)


class LinearRegressionB(BaseModel):

    def fit(self, x, y):
       X_extended = np.hstack((x, np.ones((len(x), 1))))
       self.coef_ = np.linalg.inv(np.dot(X_extended.T, X_extended)).dot(np.dot(X_extended.T, y))

    def predict(self, x):
        X_extended = np.hstack((x, np.ones((len(x),1))))
        return np.dot(X_extended,np.reshape(self.coef_, (-1)))
