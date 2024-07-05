import numpy as np
import scipy.sparse
from typing import Union


def gershgorin(A: np.array):
    """Approximate an interval that the Eigenvalues of the matrix A lie in
    by the Gershgorin circles.
    For now assume that the matrix is symmetric and thus the spectrum
    entirely real."""
    a, b = 0, 0
    n = A.shape[0]
    for i in range(n):
        center = A[i, i]
        radius = np.sum(np.abs(A[i, range(n) != i]))
        a = min(a, center - radius)
        b = max(b, center + radius)
    return a, b


def get_length_gershgorin(A: Union[np.array, scipy.sparse.sparray], stopping_acc: float):
    low, high = gershgorin(-A)
    print("For now assume the spectrum of A is entirely real and negative.")
    rho = high / 4
    # Use the first error bound from the paper
    m = int(np.sqrt(-np.log(stopping_acc / 10) * 5 * rho))
    assert np.sqrt(4 * rho) <= m
    assert m <= 2 * rho
    return m
