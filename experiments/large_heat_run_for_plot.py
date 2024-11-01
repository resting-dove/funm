import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
import subprocess
from utils import get_fig_ax, postprocess_style, Colors

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


def get_lanczos_errors(V, H, beta, exact, matfunc, step=1, upper=100, norm=scipy.linalg.norm):
    m = H.shape[1]
    idx = np.arange(0, min(m + 1, upper), step)
    errors = np.zeros(len(idx))
    errors[0] = norm(exact)

    for idxx, i in enumerate(idx[1:], start=1):
        H_exp = matfunc(H[:i, :i])
        H_exp_jax = H_exp[:, [0]]
        f = beta * (V[:, :i] @ H_exp_jax)
        errors[idxx] = norm(f.flatten() - exact)
    return idx, errors


def expm_sparse_tridiag(T: scipy.sparse.csr_array):
    w, v = scipy.linalg.eigh_tridiagonal(T.diagonal(0), T.diagonal(-1))
    return v @ np.diagflat(np.exp(w)) @ v.T


def get_lanczos_approx(V, H, beta, matfunc):
    m = H.shape[1]
    H_exp = matfunc(H[:m, :m])
    H_exp_jax = H_exp[:, [0]]
    f = beta * (V[:, :m] @ H_exp_jax)
    return f


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

    func_dense = scipy.linalg.expm
    func_sparse = scipy.sparse.linalg.expm
    func_sparse_sym = expm_sparse_tridiag
    func_scalar = np.exp
    bound_n = 550
    norm_name = "A"
    exact_n = 3375

    u0 = prepare_starting_vector3(n)


    def A_norm(x, axis=None):
        if axis is None:
            return np.sqrt(x.T @ (np.sign(evals[-1]) * (t * A)) @ x)
        elif axis == 1:
            return [A_norm(x[i, :]) for i in range(x.shape[0])]
        elif axis == 0:
            return [A_norm(x[:, i]) for i in range(x.shape[1])]
        else:
            raise RuntimeError()


    if norm_name == "2":
        norm = scipy.linalg.norm
    elif norm_name == "A":
        norm = A_norm
    else:
        raise RuntimeError()

    beta = float(np.linalg.norm(u0))
    (v, V, H, m) = arnoldi(t * A, u0.flatten() / beta, exact_n + 1, trunc=1)
    exact = get_lanczos_approx(V, H, beta, func_sparse_sym)
    exact = exact.flatten()
    fig, ax = get_fig_ax()
    colors = Colors()
    idx, lanczos_errors = get_lanczos_errors(V, scipy.sparse.csc_array(H), beta, exact, func_sparse_sym, 2,
                                             upper=400, norm=norm)
    plot_store["Lanczos idx"] = idx
    plot_store["Lanczos error_norms"] = lanczos_errors
    lanc_plot = ax.plot(idx, lanczos_errors, color='black', label=r"m=$\infty$")
    markers = ["o", "^", "x"]
    for j, krylov_size in enumerate([20, 10, 6]):
        num_starts = bound_n // krylov_size + 1

        # Calculate the matrix exponential
        npfs, npupdate_norms, final_size = lanczos_method(t * A, u0.flatten(), func_sparse,
                                                          krylov_size=krylov_size, max_starts=num_starts,
                                                          stopping_acc=-np.inf, arnoldi_acc=-np.inf,
                                                          stopping_decay=-np.inf)
        error_norms = [norm(exact.flatten())] + list(norm(exact.reshape((-1, 1)) - npfs, axis=0))
        idx = get_index(final_size, krylov_size)
        name = f"m:{krylov_size}"
        plot_store[name + " errors"] = error_norms
        plot_store[name + " update_norms"] = npupdate_norms
        plot_store[name + " idx"] = idx
        line, = ax.plot(idx, error_norms, linestyle="solid", c=colors[j])
        line, = ax.plot(idx[1:], npupdate_norms, linestyle="none", c=colors[j], marker=markers[j], label=name)

    np.savez(os.path.join(root_path, "artifacts", "plot_store" + f"large_heat{n}_{norm_name}-norm"), **plot_store)

    fig.suptitle(
        rf"${func_scalar.__name__}(A)b$, $\Lambda(A)\subset[{min(evals):.1f}, {max(evals):.1f}]$, " + r"$A\in\mathbb{"
                                                                                                      r"R}^{" + rf"{
        N}\times{N}" + "}$")
    ax.set_yscale("log")
    ax.set_ylim(bottom=np.finfo(evals[0].dtype).eps / 1000)
    ax.set_xlabel("Lanczos iterations")
    ax.set_ylabel(r"Error $||\cdot||_{" + norm_name + r"}$")
    ax.legend(framealpha=.5, scatterpoints=1, numpoints=1)
    postprocess_style()
    fig.tight_layout()
    fig.savefig(os.path.join(root_path, f"figures/large_heat_{n}_restarts_{norm_name}-norm.png"))
    plt.show()
