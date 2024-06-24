import numpy as np
import scipy


def extend_arnoldi(A, V_big: np.array, m: int, s: int, trunc=-1, reorth_num=0):
    """Extend a given Arnoldi decomposition of dimension s up to dimention m.
    """
    H = np.zeros((m + 1, m + 1))
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


def arnoldi(A, w: np.array, m: int, trunc=np.inf):
    """Calculate an Arnoldi decomposition of dimension m.
    """
    breakdown = False
    H = np.zeros((m + 1, m + 1))
    new_V_big = np.empty((w.shape[0], m))
    new_V_big[:, 0] = w
    # make the k_small column in H and the k_small+1 column in V
    for k_small in np.arange(m):
        w = new_V_big[:, k_small]
        w = A.dot(w)

        sj = max(0, k_small - trunc)  # start orthogonalizing from this index
        for j in np.arange(sj, k_small + 1):
            v = new_V_big[:, j]
            ip = np.dot(v, w)
            H[j, k_small] += ip
            w = w - ip * v
        eta = np.sqrt(np.dot(w, w))
        H[k_small + 1, k_small] = eta
        w = w / eta
        if np.abs(eta) < k_small * np.finfo(eta.dtype).eps * np.linalg.norm(H[:, k_small]):
            breakdown = True
        if k_small < m - 1:
            new_V_big[:, k_small + 1] = w
    H = H[:m + 1, :m]

    return w, new_V_big, H, breakdown
