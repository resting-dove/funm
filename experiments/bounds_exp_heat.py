import matplotlib.pyplot as plt
import scipy
import os
from utils import get_fig_ax, Colors, postprocess_style
from src.matfuncb.krylov_basis import arnoldi
from src.matfuncb.error_bounds import *

root_path = os.getcwd()


def get_index(final_size: int, krylov_size: int) -> list:
    idx = [krylov_size * i for i in range((final_size + 1) // krylov_size)]
    if idx[-1] != final_size:
        idx += [final_size]
    return idx


def prepare_starting_vector2(evecs, n: int):
    evecs = np.random.random((n ** 3, 1))
    return (evecs / scipy.linalg.norm(evecs))


def sqrtm(H: scipy.sparse.csr_array):
    return scipy.linalg.sqrtm(H.todense())


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
    n = 25  # Interior grid points in each direction
    N = n ** 3  # Total number of interior points
    h = 1 / (n + 1)
    lap = scipy.sparse.linalg.LaplacianNd((n, n, n), boundary_conditions='dirichlet')
    A = 1 / h ** 2 * lap.tosparse()
    t = 0.1
    evals = t / h ** 2 * lap.eigenvalues(N)
    print("evals gotten")

    func_dense = scipy.linalg.expm
    func_sparse = scipy.sparse.linalg.expm
    func_scalar = np.exp
    center, w = min(evals), 0
    radius = np.abs(center - w)
    bound_n = 200

    if f"evecs_heat_{n}_{t}.npy" in os.listdir(os.path.join(root_path, "precalculated")):
        evecs = np.load(os.path.join(os.path.join(root_path, "precalculated"), f"evecs_heat_{n}_{t}.npy"))
    else:
        evecs = lap.eigenvectors(N)
        np.save(os.path.join(os.path.join(root_path, "precalculated"), f"evecs_heat_{n}_{t}.npy"), evecs)
    print("evecs gotten")

    u0 = prepare_starting_vector2(evecs, n)
    print("prepared u0")
    exact = evecs.T @ u0.flatten()
    print("first matvec")
    exact = np.diagflat(func_scalar(evals)) @ exact
    exact = (evecs @ exact)
    print("calculated exact")
    evecs = None  # Maybe this frees space
    print("killed evecs")
    fig, ax = get_fig_ax()
    colors = Colors()
    i = 0
    ms, bounds = hochbruck_lubich(min(evals), 1, n=bound_n)
    ax.plot(ms, bounds, label="HL", linestyle="-", c=colors[i])

    i += 1
    ms, bounds = chen_musco(min(evals), max(evals), w=w, n=bound_n, f=func_scalar, center=center, radius=radius)
    ax.plot(ms, bounds, label="CGMM", linestyle="-", c=colors[i])

    ms, bounds = chen_musco_no_kappa(t * A, u0.flatten(), min(evals), max(evals), w=w, n=bound_n, f=func_scalar,
                                     S=[min(evals), max(evals)], center=center, radius=radius)
    ax.plot(ms, bounds, linestyle=":", c=colors.get(i, False))

    beta = np.linalg.norm(u0.flatten())
    (v, V, H, m) = arnoldi(t * A, u0.flatten() / beta, bound_n + 50, trunc=1)
    print("Arnoldi finished")
    idx, lanczos_errors = get_lanczos_errors(V, scipy.sparse.csc_array(H), beta, exact, func_sparse, 2,
                                             upper=bound_n)
    lanc_plot = ax.plot(idx, lanczos_errors, color='black')

    i += 1
    ms, bounds = chen_musco_post_no_kappa(t * A, u0.flatten(), H[:bound_n, :bound_n], w=w, center=center, radius=radius,
                                          f=func_scalar)
    ax.plot(ms, bounds, linestyle=":", c=colors.get(i, False))

    ms, bounds = chen_musco_post(H[:bound_n, :bound_n], w=w, f=func_scalar, fix_0_eval=False)
    ax.plot(ms, bounds, label="CGMM", linestyle="--", c=colors[i])

    i += 1
    ms, bounds = saad_post_for_plot(H, 10, starts=bound_n // 10 + 1, f=func_dense)
    ax.plot(ms, bounds, label="S", linestyle="--", c=colors[i])

    # np.savez(os.path.join(root_path, "artifacts", "plot_store" + f"heat{n}_bounds"), **plot_store)

    ax.set_title(
        rf"Heat equation, $\Lambda(A)\subset[{min(evals):.1f}, {max(evals):.1f}]$, " + r"$A\in\mathbb{R}^{" + rf"{N}\times{N}" + "}$")
    ax.set_yscale("log")
    ax.set_ylim(bottom=np.finfo(evals[0].dtype).eps / 1000, top=10000)
    ax.set_xlabel("Lanczos iterations")
    ax.set_ylabel(r"Error $||\cdot||_2$")
    ax.legend(framealpha=.5, scatterpoints=1, numpoints=1)
    postprocess_style()
    fig.tight_layout()
    plt.savefig(os.path.join(root_path, f"figures/bounds_exp_heat_{n}.png"))
    # fig.savefig(os.path.join(root_path, f"figures/heat{n}_bounds_ErrorPlot.png"))
    plt.show()
    1 + 1
