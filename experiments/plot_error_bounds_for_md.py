import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

from src.matfuncb.krylov_basis import arnoldi
from src.matfuncb.matfuncb import matfuncb
from test_np.utils.matrix_factory import get_symmetric_matrix_by_evals
from src.matfuncb.error_bounds import *

# Investigate how the error bounds for the Lanczos method look for the rough matrix we have in MD


def expm(A):
    return scipy.linalg.expm(A.toarray())


if __name__ == "__main__":
    evals = -1 * np.array(
        [2000, 20, 20, 20, 20] + list(np.arange(10, 1000, 100)) + list(np.arange(0.1, 10, 0.01)) + [0, 0, 0, 0, 0, 0])
    evals[40:] = 0
    n = min(len(evals), 100)
    Omega2, S = get_symmetric_matrix_by_evals(evals, True, "./precalculated", load=True, save=False)
    t = 1 / 21  # seconds
    b = np.random.random(len(evals))

    exact = S @ np.diag(np.exp(t * evals)) @ S.T @ b
    max_acc = 1e-16
    errors = []
    i_s = []
    for i in range(2, 100):  # int(len(evals) / 2)):
        app, info = matfuncb(t * Omega2, b, expm, k=i, symmetric=True)
        err = np.linalg.norm(app - exact)
        errors.append(err)
        i_s.append(i)
        if err <= max_acc:
            break
    plt.plot(i_s, errors, "x")
    plt.title(r"A priori error bounds for $\exp(-tA)$ with $||A||_2 = 2000$.")
    plt.plot(*hochbruck_lubich(evals[0], t, n), label="Hochb&Lub", linestyle="solid")
    if np.abs(evals[0]) * t <= 1:
        plt.plot(*saad(np.abs(evals[0]), t, n), label="Saad", linestyle="dashed")
    plt.plot(*ye(-evals[-1], -evals[0], t=t, n=n, alpha=0), label="Ye 0", linestyle="dotted")
    plt.plot(*ye(-evals[-1], -evals[0], t=t, n=n, alpha=0.5 * t), label="Ye 0.5t", linestyle="dotted")
    plt.plot(*ye(-evals[-1], -evals[0], t=t, n=n, alpha=t), label="Ye t", linestyle="dotted")
    plt.plot(*chen_musco(t * evals[0], t * evals[-1], w=1, n=n), label="Musco", linestyle="dashdot")
    plt.yscale("log")
    # plt.ylim(top=100, bottom=1e-24)
    plt.legend()
    plt.show()
    w = b / np.linalg.norm(b)
    (w, V, T, breakdown) = arnoldi(t * Omega2, w, 40)
    plt.plot(i_s, errors, "x")
    plt.title(r"A posteriori error bounds for $\exp(-tA)$ with $||A||_2 = 2000$.")
    # plt.scatter(*chen_musco_post(T[:3, :3], w=-100), label="Musco")
    plt.plot(*chen_musco_post(T[:10, :10], w=-100), "or", label="Musco")
    plt.plot(*chen_musco_post(T[:25, :25], w=-100), "or", label="Musco")
    plt.plot(*chen_musco_post(T[:40, :40], w=-100), "or", label="Musco")
    plt.yscale("log")
    plt.legend()
    plt.show()

    plt.plot(np.arange(50), ye_entry_1(np.arange(50), t, -evals[-1], -evals[0], q=0.4), label="q=0.4")
    plt.plot(np.arange(1000), ye_entry_1(np.arange(1000), t, -evals[-1], -evals[0], q=0.1), label="q=0.1")
    plt.plot(np.arange(1000), ye_entry_1(np.arange(1000), t, -evals[-1], -evals[0], q=0.5), label="q=0.5")
    plt.plot(np.arange(1000), ye_entry_1(np.arange(1000), t, -evals[-1], -evals[0], q=0.9), label="q=0.9")
    plt.plot(np.arange(1000), ye_entry_2(np.arange(1000), t, -evals[0]), label="Second")
    plt.yscale("log")
    # plt.ylim(top=1000, bottom=1e-32)
    plt.legend()
    plt.show()
    1 + 1
