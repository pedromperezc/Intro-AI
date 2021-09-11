from sklearn.linear_model import LinearRegression
from unittest import TestCase
from clase_3.ejercicios.models import LinearRegressionB
import numpy as np


class ModelTestCase(TestCase):

    @staticmethod
    def test_model():
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        Y = np.dot(X, np.array([1, 2])) + 3
        modelo = LinearRegressionB()
        modelo.fit(X, Y)
        predicted = modelo.predict(np.array([3, 5]))
        regressor = LinearRegression()
        regressor.fit(X, Y)
        predict = regressor.predict(np.array([[3, 5]]))

        np.testing.assert_equal(np.round(predicted), np.round(predict[0]))
