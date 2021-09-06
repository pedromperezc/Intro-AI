from unittest import TestCase
import numpy as np
from clase_2.ejercicios.src.PCA_numpy import pca_numpy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PcaTestCase(TestCase):

    def test_pca(self):
        X = np.array([[0.8, 0.7], [-0.1, 0.1]])

        v, w = pca_numpy(X, 2)

        x_std = StandardScaler(with_std=False).fit_transform(X)
        pca = PCA(n_components=2)
        pca_sci = pca.fit_transform(x_std)

        np.testing.assert_allclose(w, pca_sci, atol=1e-07)

