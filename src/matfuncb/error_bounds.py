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


def expm_error_bound(rho: float, stopping_acc: float):
    # TODO: Where does this really come from
    # TODO: Adjust to other intervals than just [0,b]
    m = int(np.sqrt(-np.log(stopping_acc / 10) * 5 * rho))
    return m


def get_length_gershgorin(A: Union[np.array, scipy.sparse.sparray], stopping_acc: float):
    low, high = gershgorin(A)
    print(f"Largest eigenvalue estimated to be {high}.")
    print("For now assume the spectrum of A is entirely real and positive.")
    rho = high / 4
    # Use the first error bound from the paper
    m = expm_error_bound(rho, stopping_acc)
    assert np.sqrt(4 * rho) <= m
    assert m <= 2 * rho
    return m


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
    high = power_method(A, b, 3)
    print(f"Largest eigenvalue estimated to be {high}.")
    print("For now assume the spectrum of A is entirely real and positive.")
    rho = high / 4
    # Use the first error bound from the paper
    m = int(np.sqrt(-np.log(stopping_acc / 10) * 5 * rho))
    assert np.sqrt(4 * rho) <= m
    assert m <= 2 * rho
    return m
