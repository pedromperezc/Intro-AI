import numpy as np

def pca_numpy(X, numero_componentes):
    X_centrado = X - X.mean(axis=0)
    cov_matrix = np.cov(X_centrado.T)
    w, v = np.linalg.eig(cov_matrix)
    v = v[np.argsort(w)[::-1]]
    proyec = X_centrado.dot(v)
    return w[:numero_componentes], proyec[:numero_componentes]