import matlab.engine
import scipy.sparse
import numpy as np


def cos_sinc(A: scipy.sparse.spdiags):
    w, v = scipy.linalg.eigh_tridiagonal(A.diagonal(), A.diagonal(-1))
    C = v @ np.diagflat(np.cos(np.sqrt(w))) @ v.T
    S = v @ np.diagflat(np.sinc(np.sqrt(w) / np.pi)) @ v.T
    return C, S


if __name__ == "__main__":
    eng = matlab.engine.start_matlab()
    eng.addpath("./wkm")
    rng = np.random.default_rng(42)
    n = 10
    d, u = rng.uniform(10, 11, (n)), rng.normal(0, 1, (n - 1))
    l = u
    A = scipy.sparse.spdiags(np.array([np.append(l, [0]), d, np.append([0], u)]), [-1, 0, 1], n, n)
    C, S = cos_sinc(A)

    C_m, S_m = eng.tridiag_wkm(A.diagonal(-1), A.diagonal(), A.diagonal(1), nargout=2)

    1 + 1
    eng.quit()
