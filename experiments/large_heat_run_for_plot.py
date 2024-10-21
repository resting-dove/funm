import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
import subprocess

from src.matfuncb.np_funm import lanczos_method
from src.matfuncb.krylov_basis import arnoldi

root_path = os.getcwd()


def get_index(final_size: int, krylov_size: int) -> list:
    idx = [krylov_size * i for i in range((final_size + 1) // krylov_size)]
    if idx[-1] != final_size:
        idx += [final_size]
    return idx


def prepare_starting_vector3(n: int):
    rng = np.random.default_rng(101)
    evecs = rng.normal(0, 100, size=(n ** 3))
    return evecs / scipy.linalg.norm(evecs)


def get_lanczos_errors(V, H, beta, exact, matfunc, step=1, upper=100):
    m = H.shape[1]
    idx = np.arange(0, min(m + 1, upper), step)
    errors = np.zeros(len(idx))
    errors[0] = np.linalg.norm(exact)

    for idxx, i in enumerate(idx[1:], start=1):
        H_exp = matfunc(H[:i, :i])
        H_exp_jax = H_exp[:, [0]]
        f = beta * (V[:, :i] @ H_exp_jax)
        errors[idxx] = np.linalg.norm(f.flatten() - exact)
    return idx, errors


if __name__ == "__main__":
    plot_store = {}
    plot_store["filename"] = os.path.basename(__file__)
    plot_store["git_commit"] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

    n = 50  # Interior grid points in each direction
    N = n ** 3  # Total number of interior points
    h = 1 / (n + 1)
    lap = scipy.sparse.linalg.LaplacianNd((n, n, n), boundary_conditions='dirichlet')
    A = 1 / h ** 2 * lap.tosparse()
    t = 0.1
    evals = t / h ** 2 * lap.eigenvalues(N)
    print("evals gotten")

    plot_store['n'] = n
    plot_store['t'] = t

    u0 = prepare_starting_vector3(n)

    fig, ax = plt.subplots()

    beta = np.linalg.norm(u0.flatten())
    exact_n = 3375
    (w, V, H, m) = arnoldi(t * A, u0.flatten() / beta, exact_n + 1, trunc=1)
    print("Arnoldi finished")
    exact = beta * (V[:, :exact_n] @ scipy.sparse.linalg.expm(scipy.sparse.csc_array(H[:exact_n, :exact_n]))[:, [0]]).flatten()
    idx, lanczos_errors = get_lanczos_errors(V, scipy.sparse.csc_array(H), beta, exact, scipy.sparse.linalg.expm, 2,
                                             upper=400)
    plot_store["Lanczos idx"] = idx
    plot_store["Lanczos error_norms"] = lanczos_errors
    lanc_plot = ax.plot(idx, lanczos_errors, color='black', label=r"m=$\infty$")

    for krylov_size in [20, 10, 6]:
        num_starts = 550 // krylov_size + 1

        # Calculate the matrix exponential
        npfs, npupdate_norms, final_size = lanczos_method(t * A, u0.flatten(), scipy.sparse.linalg.expm,
                                                          krylov_size=krylov_size, max_starts=num_starts,
                                                          stopping_acc=-np.inf, arnoldi_acc=-np.inf,
                                                          stopping_decay=-np.inf)
        error_norms = [np.linalg.norm(exact - 0)] + list(np.linalg.norm(exact.reshape((-1, 1)) - npfs, axis=0))
        idx = get_index(final_size, krylov_size)
        name = f"m:{krylov_size}"
        plot_store[name + " error_norms"] = error_norms
        plot_store[name + " idx"] = idx
        line, = ax.plot(idx, error_norms, label=name, marker=".", linestyle="None")

    np.savez(os.path.join(root_path, "artifacts", "plot_store" + f"heat{n}"), **plot_store)

    ax.set_title(f"Heat equation with n={n}")
    ax.set_yscale("log")
    ax.set_ylim(bottom=max(np.finfo(evals[0].dtype).eps, plt.ylim()[0]) / 1000)
    ax.set_xlabel("Lanczos iterations")
    ax.set_ylabel("Error")
    ax.legend(framealpha=.5, scatterpoints=1, numpoints=1)
    fig.savefig(os.path.join(root_path, f"figures/heat{n}ErrorPlot.png"))
    plt.show()
