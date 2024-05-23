import numpy as np
import scipy


def expm(A, b: np.array):
    expm_A = scipy.linalg.expm(A)
    f = expm_A @ b
    eigvals = np.log(scipy.linalg.eigvals(expm_A))

    return f, eigvals
