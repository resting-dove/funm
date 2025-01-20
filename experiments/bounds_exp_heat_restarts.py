import matplotlib.pyplot as plt
import scipy
import os
import subprocess
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


def prepare_starting_vector2(evecs, n: int, norm=scipy.linalg.norm):
    rng = np.random.default_rng(33)
    evecs = rng.random((n ** 3, 1))
    return (evecs / norm(evecs))


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


    def A_norm(x, axis=None):
        if axis is None:
            return np.sqrt(x.T @ (np.sign(evals[-1]) * t * A) @ x)
        elif axis == 1:
            return [A_norm(x[i, :]) for i in range(x.shape[0])]
        elif axis == 0:
            return [A_norm(x[:, i]) for i in range(x.shape[1])]
        else:
            raise RuntimeError()


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
        error_norms = [norm(exact - 0)] + list(norm(exact.reshape((-1, 1)) - npfs, axis=0))
        idx = get_index(final_size, krylov_size)
        name = f"m:{krylov_size}"
        plot_store[name + " errors"] = error_norms
        plot_store[name + " updates"] = npupdate_norms
        plot_store[name + " idx"] = idx
        line, = axs[j].plot(idx, error_norms, label=name, linestyle="solid", c="black")
        line, = axs[j].plot(idx[1:], npupdate_norms, linestyle="none", c="black", marker=markers[j])

        i = 1
        beta = np.linalg.norm(u0.flatten())
        (v, V, H, m) = arnoldi(t * A, u0.flatten() / beta, krylov_size + 50, trunc=1)

        i += 1
        HH, v = restarted_lanczos(t * A, u0.flatten() / beta, krylov_size=krylov_size,
                                  max_starts=bound_n // krylov_size + 1)
        ms, bounds = restarted_post_no_kappa(t * A, u0.flatten(), HH[:HH.shape[1]].todense(), w=w, center=center,
                                             radius=radius,
                                             m=krylov_size, f=func_scalar, norm=norm)
        name = f"rest post nk {krylov_size}"
        plot_store[name + " bounds"] = bounds
        plot_store[name + " ms"] = ms
        axs[j].plot(ms, bounds, linestyle=":", label="CGMM", c=colors[i])

        i += 1
        ms, bounds = afanasjew_post_for_plot(HH.todense(), v, t * A, krylov_size, starts=bound_n // krylov_size,
                                             f=func_dense, norm=norm)
        name = f"Afanasjew 1 {krylov_size}"
        plot_store[name + " bounds"] = bounds[0, :]
        plot_store[name + " ms"] = ms
        axs[j].plot(ms, bounds[0, :], label="AEEG 1", linestyle="--", c=colors[i])
        name = f"Afanasjew 2 {krylov_size}"
        plot_store[name + " bounds"] = bounds[1, :]
        plot_store[name + " ms"] = ms
        axs[j].plot(ms, bounds[1, :], label="AEEG 2", linestyle="--", c=colors[i + 1])

        axs[j].set_yscale("log")
        axs[j].set_ylim(bottom=np.finfo(evals[0].dtype).eps / 1000, top=10000)
        axs[-1].set_xlabel("Lanczos iterations")
        axs[j].set_ylabel(fr"Error $||\cdot||_{norm_name}$")
        axs[j].legend(framealpha=.5, scatterpoints=1, numpoints=1)

    np.savez(os.path.join(root_path, "artifacts", "plot_store" + f"bounds_exp_heat_{n}_restarts_{norm_name}-norm"),
             **plot_store)

    fig.suptitle(
        rf"Heat equation, $\Lambda(A)\subset[{min(evals):.1f}, {max(evals):.1f}]$, " + r"$A\in\mathbb{"
                                                                                       r"R}^{" + rf"{
        N}\times{N}" + "}$")
    postprocess_style()
    fig.tight_layout()
    # fig.savefig(os.path.join(root_path, f"figures/bounds_exp_heat_{n}_restarts_{norm_name}-norm.png"))
    plt.show()
    1 + 1
