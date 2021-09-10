from sklearn.linear_model import LinearRegression
from unittest import TestCase
from clase_3.ejercicios.models import LinearRegression_b
import numpy as np

class ModelTestCase(TestCase):

    def test_model(self):
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        Y = np.dot(X, np.array([1, 2])) + 3
        modelo = LinearRegression_b()
        modelo.fit(X, Y)
        predited = modelo.predict(np.array([3, 5]))
        regressor = LinearRegression()
        regressor.fit(X, Y)
        predict = regressor.predict(np.array([[3, 5]]))

        np.testing.assert_equal(np.round(predited), np.round(predict[0]))
