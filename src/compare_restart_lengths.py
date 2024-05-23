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
    n = 120
    n_half_imag = 4
    EWs = -np.arange(1, n + 1) / n
    EWs[-2:] = [-2000, -150]
    EWs[0] = -200
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


    exact, _, _ = np_funm_krylov(A, b, {"restart_length": n, "num_restarts": 1})
    for restart_length in [2, 4, 8, 16]:
        num_restarts = n // restart_length + 1
        param = {
            "restart_length": restart_length,
            "num_restarts": num_restarts
        }

        # Calculate the matrix exponential
        npfs, npeigvals, npupdate_norms = np_funm_krylov(A, b, param)
        npnorms = list(np.linalg.norm(exact - npfs, axis=0))
        plt.plot(restart_length * np.arange(1, param["num_restarts"] + 1), npnorms, label=f"m:{restart_length}")
        #plt.scatter(np.arange(1, param["num_restarts"] + 1), npupdate_norms, label=f"m:{restart_length}")
    plt.title("Error of restarted expm for matrix with imaginary EV")
    plt.legend(framealpha=.5)
    plt.yscale("log")
    plt.ylim(bottom=max(np.finfo(npfs.dtype).eps, plt.ylim()[0]) / 10)
    plt.xlabel("Arnoldi iterations")
    plt.show()

    plt.title("Ritz values of restarted expm for matrix with imaginary EV")
    plt.scatter(np.real(EWs), np.imag(EWs), marker="o")
    # plt.scatter(np.real(sceigvals), np.imag(sceigvals), marker="x", label="scipy", s=10)
    for k in [0, param["num_restarts"] - 1]:
        plt.scatter(np.real(npeigvals[k]), np.imag(npeigvals[k]), marker="x", label=k, s=20 - k)

    plt.legend()
    plt.ylabel("Imaginary")
    plt.xlabel("Real")
    plt.show()
    1 + 1