import pandas as pd
import numpy as np


class SetDatos(object):
    def __init__(self, ruta):
        self.data = pd.read_csv(ruta).to_numpy()[:, 1:]

    def split(self):
        indices = np.random.permutation(self.data.shape[0])
        numero_muestras_70 = int(len(self.data) * 0.7)
        numero_muestras_20 = int(len(self.data) * 0.2)

        training_idx, validation_idx, test_idx = indices[:numero_muestras_70], \
                                                 indices[numero_muestras_70:numero_muestras_70 + numero_muestras_20], \
                                                 indices[numero_muestras_70 + numero_muestras_20:]

        return self.data[training_idx, :], self.data[validation_idx, :], self.data[test_idx, :]


