import numpy as np

def get_plane_wave_basis_set(G_vectors, k_vectors):
    """
    G_vectors: np.array
    k_vectors: np.array

    return a dict (key: k, values: [PW(k, G_i) for G_i in G_vectors])
    """
    pw_set = []
    for k in k_vectors:
        tmp = []
        
        for G in G_vectors:
            pw = PlaneWave(G, k.reshape(1,-1))
            tmp.append(pw)

        pw_set.append(tmp)

    return pw_set

class PlaneWave:
    def __init__(self, G, k):
        self.G = G
        self.k = k

    def __eq__(self, pw2):
        return np.allclose(self.G, pw2.G) and np.allclose(self.k, pw2.k)

    def kinetic_integral(self, pw2):
        if self == pw2:
            return np.linalg.norm(self.k + self.G)**2
        return 0.0
