import numpy as np
from clase_2.ejercicios.src.synthetic import Syntethic
from sklearn.cluster import KMeans


def indice_cercano(dist_calculada):
    indice = np.argmin(dist_calculada, axis=0)
    return indice


def distancia_centroide(x, c):
    c = c[:, np.newaxis]
    dist_calculada = np.sqrt(np.sum((x - c) ** 2, axis=2))
    return dist_calculada


def k_means(x, n_cluster):
    idx = np.random.randint(0, len(x), n_cluster)
    centroid = x[idx]
    MAX_ITER = 10000
    threshold = 0.0001
    distancia_entre_centroides_ant = 0
    for i in range(MAX_ITER):
        centroid_ant = centroid
        centroid, indice = k_means_loop(x, centroid, n_cluster)
        distancia_entre_centroides = np.sum(distancia_centroide(centroid_ant, centroid))
        if abs(distancia_entre_centroides_ant - distancia_entre_centroides) <= threshold:
            break
        distancia_entre_centroides_ant = distancia_entre_centroides
    return centroid, indice

def k_means_loop(x, centroid, n):
    distancia = distancia_centroide(x, centroid)
    centroid_cercano = indice_cercano(distancia)
    for i in range(n):
        mask = [centroid_cercano == i]
        centroid[i] = np.mean(np.compress(mask[0], x, axis=0), axis=0)
    return centroid, centroid_cercano

n=1000
separacion = 10
data = Syntethic(n, separacion)
x, y = data.crear_cluster()
centroide, indice = k_means(x, 2)
print(centroide)

#scikit-learn
kmeans = KMeans(n_clusters=2, random_state=0).fit(x)
print(kmeans.cluster_centers_)
