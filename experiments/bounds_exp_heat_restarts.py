import matplotlib.pyplot as plt
import scipy
import os
from utils import get_fig_axs, Colors, postprocess_style
from src.matfuncb.krylov_basis import arnoldi
from src.matfuncb.error_bounds import *
from src.matfuncb.np_funm import lanczos_method, fill_block_in_top_right

root_path = os.getcwd()


def get_index(final_size: int, krylov_size: int) -> list:
    idx = [krylov_size * i for i in range((final_size + 1) // krylov_size)]
    if idx[-1] != final_size:
        idx += [final_size]
    return idx


def prepare_starting_vector2(evecs, n: int):
    rng = np.random.default_rng(42425)
    evecs = rng.normal(0, 10, (n ** 3, 1))
    # return evecs / scipy.linalg.norm(evecs)
    return (evecs / scipy.linalg.norm(evecs))


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


def restarted_lanczos(A, b: np.array, krylov_size: int = np.inf, *, max_starts: int = 1,
                      arnoldi_acc=1e-10):
    assert krylov_size > 0
    assert max_starts >= 1
    n = b.shape[0]
    beta = float(np.linalg.norm(b))
    w = b / beta
    m = krylov_size
    HH = scipy.sparse.csc_array((0, 0),
                                dtype=b.dtype)  # ((krylov_size * max_starts + 2, krylov_size * max_starts), dtype=b.dtype)
    current_size = 0
    subdiag_array = None
    for k in range(max_starts):
        (w, V, H, breakdown) = arnoldi(A=A, w=w, m=m, trunc=1, eps=arnoldi_acc)
        if breakdown:
            print("Breakdown")
            stopping_criterion = True
            m = breakdown
        HH = scipy.sparse.block_array(([HH, None], [subdiag_array, scipy.sparse.csc_array(H[:m, :m])]), format="csc")
        eta = H[m, m - 1]
        subdiag_array = fill_block_in_top_right(eta, rows=m, cols=HH.shape[1])
        current_size += m

    return HH, w


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
    center, w = min(evals), 1
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
    fig, axs = get_fig_axs(2, 1, sharex=True)
    colors = Colors()
    markers = ["o", "^"]
    for j, krylov_size in enumerate([5, 20]):
        num_starts = bound_n // krylov_size + 1

        # Calculate the matrix exponential
        npfs, npupdate_norms, final_size = lanczos_method(t * A, u0.flatten(), func_sparse,
                                                          krylov_size=krylov_size, max_starts=num_starts,
                                                          stopping_acc=-np.inf, arnoldi_acc=-np.inf,
                                                          stopping_decay=-np.inf)
        error_norms = [np.linalg.norm(exact - 0)] + list(np.linalg.norm(exact.reshape((-1, 1)) - npfs, axis=0))
        idx = get_index(final_size, krylov_size)
        name = f"m:{krylov_size}"
        line, = axs[j].plot(idx, error_norms, label=name, marker=markers[j], linestyle="solid", c="black")
        line, = axs[j].plot(idx[1:], npupdate_norms, linestyle="none", c="black", marker=markers[j])

        i = 0

        beta = np.linalg.norm(u0.flatten())
        (v, V, H, m) = arnoldi(t * A, u0.flatten() / beta, krylov_size + 50, trunc=1)

        i += 2
        ms, bounds = restarted_post_no_kappa(t * A, u0.flatten(), H[:krylov_size, :krylov_size], w=w, center=center,
                                             radius=radius,
                                             starts=bound_n // krylov_size + 1, f=func_scalar)
        axs[j].plot(ms, bounds, linestyle=":", c=colors[i])

        ms, bounds = restarted_post(H[:krylov_size, :krylov_size], w=w, starts=bound_n // krylov_size + 1,
                                    f=func_scalar,
                                    fix_0_eval=True)
        axs[j].plot(ms, bounds, label="CGMM", linestyle="--", c=colors[i])

        i += 1
        HH, v = restarted_lanczos(t * A, u0.flatten() / beta, krylov_size=krylov_size,
                                  max_starts=bound_n // krylov_size + 1)
        ms, bounds = afanasjew_post_for_plot(HH.todense(), v, t * A, krylov_size, starts=bound_n // krylov_size,
                                             f=func_dense)
        axs[j].plot(ms, bounds[0, :], label="AEEG 1", linestyle="--", c=colors[i])
        axs[j].plot(ms, bounds[1, :], label="AEEG 2", linestyle="--", c=colors[i + 1])
        axs[j].set_yscale("log")
        axs[j].set_ylim(bottom=np.finfo(evals[0].dtype).eps / 1000, top=10000)
        axs[-1].set_xlabel("Lanczos iterations")
        axs[j].set_ylabel(r"Error $||\cdot||_2$")
        axs[j].legend(framealpha=.5, scatterpoints=1, numpoints=1)

    # np.savez(os.path.join(root_path, "artifacts", "plot_store" + f"heat{n}_bounds"), **plot_store)

    fig.suptitle(
        rf"Heat equation, $\Lambda(A)\subset[{min(evals):.1f}, {max(evals):.1f}]$, " + r"$A\in\mathbb{"
                                                                                       r"R}^{" + rf"{
        N}\times{N}" + "}$")
    postprocess_style()
    fig.tight_layout()
    plt.savefig(os.path.join(root_path, f"figures/bounds_exp_heat_restarts_{n}.png"))
    # fig.savefig(os.path.join(root_path, f"figures/heat{n}_bounds_ErrorPlot.png"))
    plt.show()
    1 + 1
