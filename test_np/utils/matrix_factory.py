import os.path

import numpy as np
import scipy


def orthogonalize(M: np.array, n: int, trunc=np.inf, reo=1):
    for _ in np.arange(reo, dtype=int):
        for j in np.arange(n, dtype=int):
            w = M[:, j]
            sj = max(j - trunc, 0)
            for k in np.arange(sj, j, dtype=int):
                v = M[:, k]
                ip = np.dot(w, v)
                w = w - ip * v
                w = w / np.linalg.norm(w)
            w = w / np.linalg.norm(w)
            M[:, j] = w
    return M


def get_orthogonal_matrix(n: int):
    rng = np.random.default_rng(44)
    S = rng.random((n, n))
    S = orthogonalize(S, n, reo=3)
    return S


def get_symmetric_matrix_by_evals(evals: np.array, return_vectors=False, path_to_vectors=None, load=False, save=False):
    """Create a symmetric diagonalizable matrix with the specified eigenvalues and random eigenvectors."""
    n = len(evals)
    if load and f"ortho_vectors_{n}.npy" in os.listdir(path_to_vectors):
        S = np.load(os.path.join(path_to_vectors, f"ortho_vectors_{n}.npy"))
    else:
        S = get_orthogonal_matrix(n)
        if save and path_to_vectors:
            np.save(os.path.join(path_to_vectors, f"ortho_vectors_{n}.npy"), S)

    A = S.transpose() @ np.diag(evals) @ S
    if not return_vectors:
        return A
    else:
        return A, S


def get_matrix_by_evals(evals: np.array, multiplicities: np.array, return_vectors=False):
    """Generate a matrix with the given eigenvalues in the given multiplicities.
    The Jordan basis is a random orthogonal matrix."""
    assert len(evals) == len(multiplicities)
    n = np.sum(multiplicities)
    blocks = []
    for i in range(len(evals)):
        blocks.append(np.eye(multiplicities[i]) * evals[i] + np.diag(np.ones(multiplicities[i] - 1), 1))
    J = scipy.linalg.block_diag(*blocks)
    S = get_orthogonal_matrix(n)
    A = S.transpose() @ J @ S
    if not return_vectors:
        return A
    else:
        return A, S


if __name__ == "__main__":
    evals = list(range(100))
    A = get_symmetric_matrix_by_evals(evals)
    1 + 1
