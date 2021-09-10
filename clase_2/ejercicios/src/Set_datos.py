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
        if os.path.isfile('rating.pkl'):
            with open('rating.pkl', 'rb') as f:
                self.array = pickle.load(f)

        else:
            data = self.build_dataset(self.ruta)
            pickle.dump(data, open("rating.pkl", "wb"))
            self.array = pickle.load(open("rating.pkl", "rb"))

    def get_array(self):
        return self.array

    def build_dataset(self, path):
        structure = [('userId', np.int64),
                     ('movieId', np.int64),
                     ('rating', np.float32),
                     ('timestamp', np.int64)]

        with open(path, encoding="utf8") as data_csv:
            data_gen = ((int(line.split(',')[0]), int(line.split(',')[1]),
                         float(line.split(',')[2]), int(line.split(',')[3]))
                        for i, line in enumerate(data_csv) if i != 0)
            data = np.fromiter(data_gen, structure)
        data_csv.close()
        return data

    def get_rows(self, fila):
        return self.array[0:fila]

os.chdir("datasets")
ruta = "ratings.csv"
archivo1 = Setdatos(ruta)

