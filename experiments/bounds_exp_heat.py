import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
import subprocess
from utils import get_fig_ax, Colors, postprocess_style
from src.matfuncb.krylov_basis import arnoldi
from src.matfuncb.error_bounds import *

root_path = os.getcwd()


def get_index(final_size: int, krylov_size: int) -> list:
    idx = [krylov_size * i for i in range((final_size + 1) // krylov_size)]
    if idx[-1] != final_size:
        idx += [final_size]
    return idx


def prepare_starting_vector2(evecs, n: int, norm=scipy.linalg.norm):
    rng = np.random.default_rng(33)
    evecs = rng.random((n ** 3, 1))
    return (evecs / norm(evecs))


def sqrtm(H: scipy.sparse.csr_array):
    return scipy.linalg.sqrtm(H.todense())


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


if __name__ == "__main__":
    plot_store = {}
    plot_store["filename"] = os.path.basename(__file__)
    plot_store["git_commit"] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

    n = 25  # Interior grid points in each direction
    N = n ** 3  # Total number of interior points
    h = 1 / (n + 1)
    lap = scipy.sparse.linalg.LaplacianNd((n, n, n), boundary_conditions='dirichlet')
    A = 1 / h ** 2 * lap.tosparse()
    t = 0.1
    evals = t / h ** 2 * lap.eigenvalues(N)
    plot_store["evals"] = (min(evals), max(evals))
    print("evals gotten")


    def A_norm(x):
        return np.sqrt(x.T @ (np.sign(evals[-1]) * t * A) @ x)


    func_dense = scipy.linalg.expm
    func_sparse = scipy.sparse.linalg.expm
    func_scalar = np.exp
    center, w = min(evals), 0
    radius = np.abs(center - w)
    bound_n = 200
    norm_name = "2"
    apply_err0 = True

    if f"evecs_heat_{n}_{t}.npy" in os.listdir(os.path.join(root_path, "precalculated")):
        evecs = np.load(os.path.join(os.path.join(root_path, "precalculated"), f"evecs_heat_{n}_{t}.npy"))
    else:
        evecs = lap.eigenvectors(N)
        np.save(os.path.join(os.path.join(root_path, "precalculated"), f"evecs_heat_{n}_{t}.npy"), evecs)
    print("evecs gotten")
    if norm_name == "2":
        norm = scipy.linalg.norm
    elif norm_name == "A":
        norm = A_norm
    else:
        raise RuntimeError()

    u0 = prepare_starting_vector2(evecs, n)
    print("prepared u0")
    exact = evecs.T @ u0.flatten()
    print("first matvec")
    exact = np.diagflat(func_scalar(evals)) @ exact
    exact = (evecs @ exact)
    print("calculated exact")
    evecs = None  # Maybe this frees space
    print("killed evecs")
    if apply_err0:
        exact_norm = norm(exact)
    else:
        exact_norm = 1
        print(f"Exact norm: {norm(exact)}")
    plot_store["exact norm"] = exact_norm
    lin_error_0 = A_norm(scipy.sparse.linalg.spsolve(t * A - w * scipy.sparse.eye(*A.shape), u0))
    print(f"Linear error 0: {lin_error_0}")
    plot_store["lin_error_0"] = lin_error_0

    fig, ax = get_fig_ax()
    colors = Colors()
    i = 0
    ms, bounds = hochbruck_lubich(min(evals), 1, n=bound_n)
    name = f"HL"
    plot_store[name + " bounds"] = bounds
    plot_store[name + " ms"] = ms
    ax.plot(ms, bounds, label="HL", linestyle="-", c=colors[i])

    i += 1
    ms, bounds = chen_musco(min(evals), max(evals), w=w, n=bound_n, f=func_scalar, center=center, radius=radius)
    name = f"CGMM prio"
    plot_store[name + " bounds"] = bounds
    plot_store[name + " ms"] = ms
    ax.plot(ms, lin_error_0 * bounds, label="CGMM", linestyle="-", c=colors[i])

    ms, bounds = chen_musco_no_kappa(t * A, u0.flatten(), min(evals), max(evals), w=w, n=bound_n, f=func_scalar, center=center,
                            radius=radius, norm=norm)
    name = f"CGMM prio nk"
    plot_store[name + " bounds"] = bounds
    plot_store[name + " ms"] = ms
    ax.plot(ms, bounds, linestyle=":", c=colors.get(i, False))

    beta = np.linalg.norm(u0.flatten())
    (v, V, H, m) = arnoldi(t * A, u0.flatten() / beta, bound_n + 50, trunc=1)
    print("Arnoldi finished")
    idx, lanczos_errors = get_lanczos_errors(V, scipy.sparse.csc_array(H), beta, exact, func_sparse, 2,
                                             upper=bound_n, norm=norm)
    name = f"Lanczos"
    plot_store[name + " errors"] = lanczos_errors
    plot_store[name + " ids"] = idx
    lanc_plot = ax.plot(idx, lanczos_errors, color='black')

    i += 1
    ms, bounds = chen_musco_post_no_kappa(t * A, u0.flatten(), H[:bound_n, :bound_n], w=w, center=center, radius=radius,
                                          f=func_scalar, norm=norm)
    name = f"CGMM post nk"
    plot_store[name + " bounds"] = bounds
    plot_store[name + " ms"] = ms
    ax.plot(ms, bounds, linestyle=":", c=colors.get(i, False))

    ms, bounds = chen_musco_post(H[:bound_n, :bound_n], w=w, f=func_scalar, fix_0_eval=False)
    name = f"CGMM post"
    plot_store[name + " bounds"] = bounds
    plot_store[name + " ms"] = ms
    ax.plot(ms, exact_norm * bounds, label="CGMM", linestyle="--", c=colors[i])

    i += 1
    ms, bounds = saad_post_for_plot(H, 10, starts=bound_n // 10 + 1, f=func_dense)
    name = f"Saad post"
    plot_store[name + " bounds"] = bounds
    plot_store[name + " ms"] = ms
    ax.plot(ms, bounds, label="S", linestyle="--", c=colors[i])

    np.savez(os.path.join(root_path, "artifacts", "plot_store" + f"_bounds_exp_heat_{n}_{norm_name}-norm"),
             **plot_store)

    ax.set_title(
        rf"Heat equation, $\Lambda(A)\subset[{min(evals):.1f}, {max(evals):.1f}]$, " + r"$A\in\mathbb{R}^{" + rf"{N}\times{N}" + "}$")
    ax.set_yscale("log")
    ax.set_ylim(bottom=np.finfo(evals[0].dtype).eps / 1000, top=10000)
    ax.set_xlabel("Lanczos iterations")
    ax.set_ylabel(fr"Error $||\cdot||_{norm_name}$")
    ax.legend(framealpha=.5, scatterpoints=1, numpoints=1)
    postprocess_style()
    fig.tight_layout()
    plt.savefig(os.path.join(root_path, f"figures/bounds_exp_heat_{n}_{norm_name}-norm.png"))
    # fig.savefig(os.path.join(root_path, f"figures/heat{n}_bounds_ErrorPlot.png"))
    plt.show()
    1 + 1
