import numpy as np
from sklearn.linear_model import LinearRegression

class BaseModel(object):

    def __init__(self):
        self.model = None

    def fit(self, x, y):
        return NotImplemented

    def predict(self, x):
        return NotImplemented


class ConstantModel(BaseModel):

    def fit(self, x, y):
        self.model = Y.mean()

    def predict(self, x):
        return np.ones(len(x)) * self.model


class LinearRegression_np(BaseModel):

    def fit(self, x, y):
        if len(X.shape) == 1:
            self.model = np.dot(x.T, y) / np.dot(x.T, x)
        else:
            self.model = np.linalg.inv(np.dot(x.T, x)).dot(np.dot(x.T, y))
    def predict(self, x):
        if len(X.shape) == 1:
            return self.model * x
        else:
            return self.model.dot(x)

class LinearRegression_b(BaseModel):

    def fit(self, x, y):
        if len(X.shape) == 1:
            self.model = np.dot(x.T, y) / np.dot(x.T, x)
        else:
            X_extended = np.hstack((x, np.ones((len(x), 1))))
            self.model = np.linalg.inv(np.dot(X_extended.T, X_extended)).dot(np.dot(X_extended.T, y))

    def predict(self, x):
        if len(X.shape) == 1:
            return self.model * x
        else:
            X_extended = np.hstack((x, 1))
            return self.model.dot(X_extended.T)

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
Y = np.dot(X, np.array([1, 2])) + 3

modelo = LinearRegression_b()

modelo.fit(X, Y)

print(modelo.predict(np.array([3,5])))

regressor = LinearRegression()
regressor.fit(X,Y)
predict = regressor.predict(np.array([[3,5]]))
print (predict)
print (regressor.coef_)
print (regressor.intercept_)

