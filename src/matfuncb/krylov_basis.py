import numpy as np
import scipy
from typing import Union


def arnoldi_step(A: Union[np.array, scipy.sparse.sparray], V: np.array, H: np.array, s: int, trunc=-1):
    """Extend a given Arnoldi decomposition of dimension s by one step.
    """
    w = V[:, s]
    w = A.dot(w)
    sj = max(0, s - trunc)  # start orthogonalizing from this index
    for j in np.arange(sj, s + 1):
        v = V[:, j]
        ip = np.dot(v, w)
        H[j, s] += ip
        w = w - ip * v
    eta = np.sqrt(np.dot(w, w))
    H[s + 1, s] = eta
    w = w / eta
    return w, V, H


def extend_arnoldi(A: Union[np.array, scipy.sparse.sparray], V: np.array, H: np.array, s: int, m: int, trunc=-1):
    """Extend a given Arnoldi decomposition from size s to size m."""
    breakdown = False
    # make the k_small column in H and the k_small+1 column in V
    for k_small in np.arange(s, m):
        w, new_V_big, H = arnoldi_step(A, V, H, k_small, trunc)
        eta = H[k_small + 1, k_small]
        if np.abs(eta) < k_small * np.finfo(eta.dtype).eps * np.linalg.norm(H[:, k_small]):
            breakdown = True
        if k_small < m - 1:
            new_V_big[:, k_small + 1] = w
        if breakdown:
            break
    H = H[:m + 1, :m]

    return w, V, H, breakdown


def arnoldi(A, w: np.array, m: int, trunc=np.inf):
    """Calculate an Arnoldi decomposition of dimension m.
    """
    breakdown = False
    H = np.zeros((m + 1, m + 1))
    new_V_big = np.empty((w.shape[0], m))
    new_V_big[:, 0] = w
    # make the k_small column in H and the k_small+1 column in V
    for k_small in np.arange(m):
        w, new_V_big, H = arnoldi_step(A, new_V_big, H, k_small, trunc)
        eta = H[k_small + 1, k_small]
        if np.abs(eta) < k_small * np.finfo(eta.dtype).eps * np.linalg.norm(H[:, k_small]):
            breakdown = True
        if k_small < m - 1:
            new_V_big[:, k_small + 1] = w
        if breakdown:
            break
    H = H[:m + 1, :m]

    return w, new_V_big, H, breakdown
