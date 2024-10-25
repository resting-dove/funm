import matplotlib.pyplot as plt
import scipy
import os
import numpy as np
import subprocess
from utils import get_fig_ax, Colors, postprocess_style
from src.matfuncb.error_bounds import get_cg_errors, get_cg_bound, get_restarted_cg_errors, get_restarted_cg_bound, \
    setup_circle_and_kappa

root_path = os.getcwd()


def prepare_starting_vector2(evecs, n: int):
    evecs = np.random.random((n ** 3, 1))
    return (evecs / scipy.linalg.norm(evecs))


if __name__ == "__main__":
    plot_store = {}
    plot_store["filename"] = os.path.basename(__file__)
    plot_store["git_commit"] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

    n = 10  # Interior grid points in each direction
    N = n ** 3  # Total number of interior points
    h = 1
    rng = np.random.default_rng(33)
    evals = 400 * (rng.random((N)) - 1)
    A = scipy.sparse.dia_array(([evals], [0]), shape=(N, N))
    t = 1
    print("evals gotten")

    evecs = scipy.sparse.eye(N)
    bound_n = 200

    plot_store['n'] = n
    plot_store['t'] = t

    u0 = prepare_starting_vector2(evecs, n)


    def A_norm(x):
        return np.sqrt(x.T @ -A @ x)


    u0 = u0 / A_norm(u0)
    exact = scipy.sparse.dia_array(([1 / evals], [0]), shape=(N, N)) @ u0.flatten()
    exact_norm = A_norm(exact)
    fig, ax = get_fig_ax(factor=0.6)
    colors = Colors()

    errors = get_cg_errors(t * A, 0, u0.flatten(), bound_n, solution=exact, norm=A_norm)
    idx = np.arange(bound_n + 1)
    name = f"m:infty"
    plot_store[name + " errors"] = errors
    plot_store[name + " idx"] = idx
    plt.plot(idx, errors, linestyle='-', color="black", label=rf"m:$\infty$")
    bottom = min(errors[errors > 0])

    _, _, kappa = setup_circle_and_kappa(min(evals), max(evals), 0)
    bounds = get_cg_bound(kappa, idx[1:])
    bounds = exact_norm * bounds
    plot_store[name + " bounds"] = bounds
    plt.plot(idx[1:], bounds, linestyle='--', color="black")

    for i, m in enumerate([5, 20]):
        starts = bound_n // m
        errors = get_restarted_cg_errors(t * A, 0, u0.flatten(), m, starts, solution=exact, norm=A_norm)
        idx = np.arange(0, starts * m + 1, m)
        name = f"m:{m}"
        plot_store[name + " errors"] = errors
        plot_store[name + " idx"] = idx
        plt.plot(idx, errors, linestyle='-', color=colors[i], label=f"m:{m}")

        bounds = get_restarted_cg_bound(kappa, m, starts)
        bounds = exact_norm * bounds
        plot_store[name + " bounds"] = bounds
        plt.plot(idx[1:], bounds, linestyle='--', color=colors[i])
    np.savez(os.path.join(root_path, "artifacts", "plot_store" + f"cg_bounds"), **plot_store)

    ax.set_yscale("log")
    ax.set_ylim(bottom=bottom / 10, top=10)
    ax.set_xlabel("Steps")
    ax.set_ylabel(r"Error $||\cdot||_A$")
    ax.legend(framealpha=.5, scatterpoints=1, numpoints=1)
    postprocess_style()
    fig.tight_layout()
    # plt.savefig(os.path.join(root_path, f"figures/cg_bounds.png"))
    fig.savefig(os.path.join(root_path, f"figures/cg_bounds.png"))
    plt.show()
