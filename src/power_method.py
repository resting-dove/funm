import numpy as np


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
