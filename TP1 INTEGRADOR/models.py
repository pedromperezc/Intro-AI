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


def cost_entropy(Y, A, n):
    return np.mean(-Y * (np.log(A)) - (1 - Y) * np.log(1 - A))


def initialize(dim):
    w = np.random.randn(dim, 1)
    b = 0
    return w, b


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def momentum_update(W, b, dw, db, states, lr, momentum_rate):
    # hyper-param typical values: learning_rate=0.01, momentum=0.9
    W_ant = W
    b_ant = b
    W = W + momentum_rate * states[0] - lr * dw
    b = b + momentum_rate * states[1] - lr * db
    states = [W - W_ant, b - b_ant]
    return W, b, states


class LogisticRegressionNumpy(BaseModel):

    def fit(self, X_train, y_train, lr, batch, epochs, momentum_rate):
        w, b = initialize(X_train.shape[1])

        n = X_train.shape[0]
        m = X_train.shape[1]
        states = np.zeros((m, 1))

        # Estandarizo los datos
        X_train = (X_train - np.mean(X_train)) / np.std(X_train)

        for i in range(epochs):
            idx = np.random.permutation(X_train.shape[0])
            X_train = X_train[idx]
            y_train = y_train[idx]

            batch_size = int(len(X_train) / batch)

            for i in range(0, len(X_train), batch_size):
                end = i + batch_size if i + batch_size <= len(X_train) else len(X_train)
                batch_X = X_train[i: end]
                batch_y = y_train[i: end]

                z = np.dot(batch_X, w) + b
                A = sigmoid(z)

                # np.sum(error * batch_X, axis=0)

                cost = cost_entropy(batch_y, A, n)

                dz = A - batch_y
                dw = (1 / batch_size) * np.dot(batch_X.T, dz)
                db = (1 / batch_size) * np.sum(dz)

                if momentum_rate > 0:
                    w, b, states = momentum_update(w, b, dw, db, states, lr, momentum_rate)
                else:
                    w = w - lr * dw
                    b = b - lr * db
        self.coef_ = w
        self.intercept_ = b

    def predict(self, X):
        p = sigmoid(X @ self.coef_ + self.intercept_)
        mask_true = p >= 0.5
        mask_false = p < 0.5
        p[mask_true] = 1
        p[mask_false] = 0
        return p
