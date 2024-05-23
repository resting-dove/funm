import scipy

from jax_funm import funm_krylov as jax_funm_krylov
from np_funm import funm_krylov as np_funm_krylov
from scipy_expm import expm
import numpy as np
import jax.numpy as jnp
from numpy import random
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    # Set up a diagonal matrix
    n = 100
    n_half_imag = 2
    EWs = -np.arange(1, n + 1) / n
    imag_sizes = np.arange(1, n_half_imag + 1)
    imag_blocks = [np.array([[EWs[i], -imag_sizes[i]], [imag_sizes[i], EWs[i]]]) for i in range(n_half_imag)]
    M = scipy.linalg.block_diag(*(imag_blocks + [ev for ev in EWs[2 * n_half_imag:]]))
    EWs = EWs.astype(np.complex64)
    EWs[:n_half_imag] += 1j * imag_sizes
    EWs[n_half_imag: 2 * n_half_imag] = EWs[:n_half_imag] - 2j * imag_sizes


    rng = np.random.default_rng(44)
    S = rng.random((n, n))
    S = orthogonalize(S, n, reo=3)
    print(f"Orthogonality of S: {jnp.linalg.norm(S.transpose() @ S - jnp.eye(n, n))}")

    A = S.transpose() @ M @ S
    b = np.ones(n) / np.linalg.norm(np.ones(n))

    param = {
        "restart_length": 4,
        "num_restarts": 25
    }

    # Calculate the matrix exponential
    fs, eigvals, update_norms = jax_funm_krylov(jnp.array(A), jnp.array(b), param)
    npfs, npeigvals, npupdate_norms = np_funm_krylov(A, b, param)
    scf, sceigvals = expm(A, b)

    exact, _, _ = np_funm_krylov(A, b, {"restart_length": n, "num_restarts": 1})

    norms = [np.linalg.norm(exact - 0)] + list(np.linalg.norm(exact - fs, axis=0))
    npnorms = [np.linalg.norm(exact - 0)] + list(np.linalg.norm(exact - npfs, axis=0))
    scnorm = np.linalg.norm(exact - scf)

    print(f"Error of scipy expm: {scnorm}")

    plt.plot(np.arange(param["num_restarts"] + 1), norms, label="jax")
    plt.plot(np.arange(param["num_restarts"] + 1), npnorms, label="numpy")
    plt.scatter(np.arange(1, param["num_restarts"] + 1), update_norms, label="jax update norms")
    plt.scatter(np.arange(1, param["num_restarts"] + 1), npupdate_norms, label="numpy update norms")
    plt.title("Error of restarted expm for matrix with imaginary EV")
    plt.legend(framealpha=.5)
    plt.yscale("log")
    plt.ylim(bottom=max(np.finfo(npfs.dtype).eps, plt.ylim()[0]) / 10)
    plt.xlabel("Restarts")
    plt.show()

    plt.title("Ritz values of restarted expm for matrix with imaginary EV")
    plt.scatter(np.real(EWs), np.imag(EWs), marker="o")
    # plt.scatter(np.real(sceigvals), np.imag(sceigvals), marker="x", label="scipy", s=10)
    for k in [0, param["num_restarts"] - 1]:
        plt.scatter(np.real(eigvals[k]), np.imag(eigvals[k]), marker="x", label=k, s=20 - k)

    plt.legend()
    plt.ylabel("Imaginary")
    plt.xlabel("Real")
    plt.show()
    1 + 1