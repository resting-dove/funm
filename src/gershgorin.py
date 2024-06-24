import numpy as np

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