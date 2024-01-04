import numpy as np

def get_k_points(X_BZ, G=None, sampling_name = "baldereschi", n = 1):
    """return k-sampling of the Brillouin zone"""
    if sampling_name == "baldereschi":
        return np.mean(X_BZ, axis=0, keepdims=True)
    elif sampling_name == "monkhorst_pack":
        n1 = n[0]
        n2 = n[1]

        b1, b2 = G[0:1,:], G[1:2,:]
            
        uu_1 = np.arange(n1).reshape(-1,1)
        uu_2 = np.arange(n2).reshape(-1,1)

        u1 = (2* uu_1 - n1 - 1) / (2 * n1)
        u2 = (2* uu_2 - n2 - 1) / (2 * n2)

        X_k = u1 * b1 + u2 * b2
        return X_k
    else:
        raise ValueError(f"Invalid sampling method {sampling_name} or n {n}")

def get_G_vectors(G, E_cutoff):
    b1 = G[0:1,:]
    b2 = G[1:2,:]
    b3 = b1 + b2
    b4 = np.asarray(-b1)
    b5 = np.asarray(-b3)
    b6 = np.asarray(-b2)

    G_list = []
    G_list.append(np.zeros((1,2)))

    i = 1
    while np.sum((i*b1)*(i*b1))/2<E_cutoff:
        G_list.append(i*b1)
        G_list.append(i*b2)
        G_list.append(i*b3)
        G_list.append(i*b4)
        G_list.append(i*b5)
        G_list.append(i*b6)
        i += 1

    return np.concatenate(G_list, axis=0)
