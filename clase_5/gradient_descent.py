import numpy as np


def costfunction(theta, X, y):
    m = np.size(y)
    theta = theta.reshape(-1, 1)

    # Cost function in vectorized form
    h = X @ theta
    J = float((1. / (2 * m)) * (h - y).T @ (h - y));
    return J;

def momentum_update(W, grads, states, hyper_param, lr):
    # hyper-param typical values: learning_rate=0.01, momentum=0.9

    W_ant = W
    W = W + hyper_param * states - lr * grads

    states = W - W_antt
    return W, states


def gradient_descent(X_train, y_train, lr=0.01, amt_epochs=100, momentum_rate=0):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    n = X_train.shape[0]
    m = X_train.shape[1]

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)
    states = np.zeros((m,1))

    for i in range(amt_epochs):
        prediction = np.dot(X_train, W)  # nx1
        error = y_train - prediction  # nx1

        grad_sum = np.sum(error * X_train, axis=0)
        grad_mul = -2/(n) * grad_sum  # 1xm
        gradient = np.transpose(grad_mul).reshape(-1, 1)  # mx1

        if momentum_rate > 0:
            W, states = momentum_update(W, gradient, states, momentum_rate, lr)
        else:
            W = W - (lr * gradient)
    return W


def stochastic_gradient_descent(X_train, y_train, lr=0.01, amt_epochs=100, momentum_rate=0):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    n = X_train.shape[0]
    m = X_train.shape[1]

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)
    states = np.zeros((m,1))

    for i in range(amt_epochs):
        idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[idx]
        y_train = y_train[idx]

        for j in range(n):
            prediction = np.matmul(X_train[j].reshape(1, -1), W)  # 1x1
            error = y_train[j] - prediction  # 1x1

            grad_sum = error * X_train[j]
            grad_mul = -2/n * grad_sum  # 2x1
            gradient = np.transpose(grad_mul).reshape(-1, 1)  # 2x1

            if momentum_rate > 0:
                W, states = momentum_update(W, gradient, states, momentum_rate, lr)
            else:
                W = W - (lr * gradient)
    return W 


def mini_batch_gradient_descent(X_train, y_train, lr=0.01, amt_epochs=100,momentum_rate=0):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    b = 16
    n = X_train.shape[0]
    m = X_train.shape[1]
    states = np.zeros((m,1))

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)

    for i in range(amt_epochs):
        idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[idx]
        y_train = y_train[idx]

        batch_size = int(len(X_train) / b)
        for i in range(0, len(X_train), batch_size):
            end = i + batch_size if i + batch_size <= len(X_train) else len(X_train)
            batch_X = X_train[i: end]
            batch_y = y_train[i: end]

            prediction = np.matmul(batch_X, W)  # nx1
            error = batch_y - prediction  # nx1

            grad_sum = np.sum(error * batch_X, axis=0)
            grad_mul = -2/n * grad_sum  # 1xm
            gradient = np.transpose(grad_mul).reshape(-1, 1)  # mx1

            if momentum_rate > 0:
                W, states = momentum_update(W, gradient, states, momentum_rate, lr)
            else:
                W = W - (lr * gradient)
    return W