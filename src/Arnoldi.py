import numpy as np


def extend_arnoldi(A, V_big: np.array, m: int, H: np.array, s: int, trunc=-1, reorth_num=0):
    """Extend a given Arnoldi decomposition (V_big, H) of dimension s up to dimention m.
    Assume that H is at least of size (m+1, m+1)."""
    eps = 1e-15
    trunc = trunc if trunc >= 0 else m - s
    breakdown = False
    w = V_big[:, s]

    # make the k column in H_full and the k+1 column in V
    # this is the k - s column in H
    for k in np.arange(s, m):
        k_small = k - s
        w = V_big[:, k]
        w = np.dot(A, w)

        sj = max([s, k - trunc])  # start orthogonalizing from this index
        for j in np.arange(sj, k + 1):
            v = V_big[:, j]
            ip = np.dot(v, w)
            H[j - s, k_small] = ip
            w = w - ip * v
        w2 = np.dot(w, w)
        H[k_small + 1, k_small] = np.sqrt(w2)

        if H[k_small + 1, k_small] < k * eps:
            breakdown = True
            print("breakdown")

        w = w / H[k_small + 1, k_small]
        if k < m:
            V_big[:, k + 1] = w

    h = H[m - s, m - s - 1]
    H = H[:m - s, :m - s]
    return w, V_big, H, h, breakdown
