import numpy as np
import seaborn as sns
from sklearn import datasets

class Syntethic(object):
    def __init__(self,n_muestra,shifted):
        self.n_muestra = n_muestra
        self.shifted = shifted

    def crear_cluster(self):
        centroid = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        centroid_shifted = centroid * self.shifted
        cluster = np.repeat(centroid, self.n_muestra/2, axis=0)
        gauss_noise = np.random.normal(loc=0,scale=2,size=[self.n_muestra,4])
        cluster_noised = cluster + gauss_noise
        cluster_ids = np.array([[0], [1]])
        cluster_ids = np.repeat(cluster_ids, self.n_muestra / 2, axis=0)
        return cluster_noised, cluster_ids


