import pandas as pd
import numpy as np
import pickle
import os


class Setdatos(object):
    instance = None

    def __new__(cls, *args, **kwargs):
        if Setdatos.instance is None:
            print("creando la instancia")
            Setdatos.instance = super().__new__(cls)
            return Setdatos.instance
        else:
            return Setdatos.instance

    def __init__(self, ruta):
        self.ruta = ruta
        if os.path.isfile('archivo.pkl'):
            with open('archivo.pkl', 'rb') as f:
                self.array = pickle.load(f)

        else:
            archivo = pd.read_csv(self.ruta, sep=",")
            self.array = np.array(archivo.to_numpy(),
                                  dtype=[('userId', np.int64), ('movieId', np.int64), ('rating', np.float64),
                                         ('timestamp', np.int64)])
            with open('archivo.pkl', 'wb') as f:
                pickle.dump(self.array, f)

    def get_array(self):
        return self.array


os.chdir("datasets")
ruta = "ratings.csv"
archivo1 = Setdatos(ruta)
print(archivo1.get_array())
