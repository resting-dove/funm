import numpy as np
import scipy


def eigs_2x2(A):
    """Source:
    https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_EigenProblem2.html#QR-Algorithm-for-computing-eigenvalues
    """
    b = -(A[-1, -1] + A[-2, -2])
    c = A[-1, -1] * A[-2, -2] - A[-2, -1] * A[-1, -2]
    d = np.sqrt(b ** 2 - 4 * c)

    if b > 0:
        return (-2 * c / (b + d), -(b + d) / 2)
    else:
        return ((d - b) / 2, 2 * c / (d - b))


def get_eigvals_qr(H, max_iterations=1000, tol=1e-6):
    """Calculate the eigenvalues of the matrix H using the QR algorithm.
    Influenced by
    https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_EigenProblem2.html#QR-Algorithm-for-computing-eigenvalues."""
    n = H.shape[0]
    eigvals = np.zeros(n)
    q, r = scipy.linalg.qr(H)
    A = np.dot(r, q)
    I = np.eye(n)
    for _ in range(max_iterations):
        if np.abs(A[-1, -2]) < tol:
            n -= 1
            eigvals[n] = A[-1, -1]
            A = A[:-1, :-1]
            I = np.eye(n)
        if n == 2:
            eigvals[:2] = eigs_2x2(A)
            break
        shift = A[-1, -1]
        q, r = scipy.linalg.qr(A - shift * I)
        A = np.dot(r, q) + shift * I
    #print(f"It took {i} iterations to get max entry to {np.max(np.tril(A))}.")
    return eigvals