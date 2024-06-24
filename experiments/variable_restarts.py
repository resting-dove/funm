import scipy
from src.np_funm import funm_krylov_v2, gershgorin_adaptive_expm, power_adaptive_expm
import numpy as np
import jax.numpy as jnp
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
    # Set up a diagonal matrix
    n = 100
    EWs = -np.arange(1, n + 1) / n
    EWs[0:4] = [-2000, -21, -20.99, 0]
    rng = np.random.default_rng(44)
    S = rng.random((n, n))
    S = orthogonalize(S, n, reo=3)
    print(f"Orthogonality of S: {jnp.linalg.norm(S.transpose() @ S - jnp.eye(n, n))}")

    A = S.transpose() @ np.diag(EWs) @ S
    b = np.ones(n) / np.linalg.norm(np.ones(n))


    exact, _, _ = funm_krylov_v2(A, b, {"restart_length": n, "num_restarts": 1})

    f, _, _, m = gershgorin_adaptive_expm(A, b, stopping_acc=1e-10)
    npnorms = [np.linalg.norm(exact)] + list(np.linalg.norm(exact - f, axis=0))
    if isinstance(m, list):
        m = [0] + m
    else:
        m = [0, m]
    plt.plot(np.cumsum(m), npnorms, label=f"Gershgorin with m={m[-1]}")

    f, _, _, m = power_adaptive_expm(A, b, stopping_acc=1e-10)
    npnorms = [np.linalg.norm(exact)] + list(np.linalg.norm(exact - f, axis=0))
    if isinstance(m, list):
        m = [0] + m
    else:
        m = [0, m]
    plt.plot(np.cumsum(m), npnorms, label=f"Gershgorin with m={m[-1]}")

    plt.title("Error of adaptive size Krylov method expm for symmetric matrix")
    plt.legend(framealpha=.5)
    plt.yscale("log")
    plt.ylim(bottom=max(np.finfo(f.dtype).eps, plt.ylim()[0]) / 10)
    plt.xlabel("Arnoldi iterations")
    plt.show()

    plt.title("Ritz values of restarted expm for matrix with imaginary EV")
    plt.scatter(np.real(EWs), np.imag(EWs), marker="o")
    # plt.scatter(np.real(sceigvals), np.imag(sceigvals), marker="x", label="scipy", s=10)
    # for k in [0, ms[-1]]:
    #     plt.scatter(np.real(npeigvals[k]), np.imag(npeigvals[k]), marker="x", label=k, s=20 - k)

    plt.legend()
    plt.ylabel("Imaginary")
    plt.xlabel("Real")
    plt.show()
    1 + 1