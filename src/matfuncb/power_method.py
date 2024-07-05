import numpy as np
from typing import Union

import scipy.sparse


def power_method(A: np.array, b: np.array, iterations: int, tol=1e-14):
    """Perform a certain number of iterations of the power method."""
    converged = False
    i = 0
    prev = 0
    nextb = A.dot(b)
    while not converged and i < iterations:
        i += 1
        b = nextb
        b = b / np.linalg.norm(b)
        nextb = A.dot(b)
        eig = b.T @ nextb / np.linalg.norm(b) ** 2
        if np.abs(eig - prev) < tol:
            break
        prev = eig
    return eig


def get_length_power(A: Union[np.array, scipy.sparse.sparray], b: np.array, stopping_acc: float):
    high = power_method(-A, b, 3)
    print("For now assume the spectrum of A is entirely real and negative.")
    rho = high / 4
    # Use the first error bound from the paper
    m = int(np.sqrt(-np.log(stopping_acc / 10) * 5 * rho))
    assert np.sqrt(4 * rho) <= m
    assert m <= 2 * rho
    return m
