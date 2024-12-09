import os
import subprocess

import matplotlib.pyplot as plt
import openmm.unit as unit
import scipy

from experiments.utils import get_fig_ax, Colors, postprocess_style
from pywkm.wkm import wkm
from semi_md.ase_md.unit_helpers import ase_unit_system
from src.matfuncb.error_bounds import *
from src.matfuncb.krylov_basis import arnoldi
from src.matfuncb.np_funm import lanczos_method, fill_block_in_top_right

# import ase.units as ase_units

root_path = os.getcwd()


def get_index(final_size: int, krylov_size: int) -> list:
    idx = [krylov_size * i for i in range((final_size + 1) // krylov_size)]
    if idx[-1] != final_size:
        idx += [final_size]
    return idx


def prepare_starting_vector2(evecs, n: int, norm=scipy.linalg.norm):
    evecs = np.random.random((n ** 3, 1))
    return (evecs / norm(evecs))


def sinc_sqrtm(H: np.ndarray):
    C, S = wkm(-H, return_sinhc=True)
    return S


def sinc_sqrt(x: float):
    # return np.sinc(np.sqrt(np.clip(x, 0, np.infty)) / np.pi)
    return np.real(np.sinc(np.sqrt(x + 0j) / np.pi))


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


def get_lanczos_approx(V, H, beta, matfunc):
    m = H.shape[1]
    H_exp = matfunc(H[:m, :m])
    H_exp_jax = H_exp[:, [0]]
    f = beta * (V[:, :m] @ H_exp_jax)
    return f


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

    omega2_0 = scipy.sparse.load_npz("Omega2_0.0femtosecond.npz")
    xi = np.load("vectors_0.0femtosecond.npz")["g_xi"]
    N = omega2_0.shape[0]

    time_step = 1 * unit.femtosecond
    # ase_time_step = time_step.value_in_unit(unit.femtosecond) * ase_units.fs
    # t = ase_time_step
    t = time_step.value_in_unit_system(ase_unit_system)

    evals = scipy.sparse.linalg.eigsh(t ** 2 * omega2_0, return_eigenvectors=False, which="BE", k=5)
    print("evals: ", evals)

    func_dense = sinc_sqrtm
    func_sparse = sinc_sqrtm
    func_sparse_sym = sinc_sqrtm
    func_scalar = sinc_sqrt
    center, w = max(evals), min(-0.1, t ** 2 * 2 * min(evals))
    radius = np.abs(center - w)
    bound_n = 100
    norm_name = "A-wI"
    apply_err0 = True
    exact_n = 2000


    def A_norm(x, axis=None):
        if axis is None:
            return np.sqrt(x.T @ (t ** 2 * omega2_0) @ x)
        elif axis == 1:
            return [A_norm(x[i, :]) for i in range(x.shape[0])]
        elif axis == 0:
            return [A_norm(x[:, i]) for i in range(x.shape[1])]
        else:
            raise RuntimeError()


    def A_wI_norm(x, axis=None):
        if axis is None:
            return np.sqrt(x.T @ (t ** 2 * omega2_0 - w * scipy.sparse.eye(*omega2_0.shape)) @ x)
        elif axis == 1:
            return [A_wI_norm(x[i, :]) for i in range(x.shape[0])]
        elif axis == 0:
            return [A_wI_norm(x[:, i]) for i in range(x.shape[1])]
        else:
            raise RuntimeError()


    if norm_name == "2":
        norm = scipy.linalg.norm
    elif norm_name == "A":
        norm = A_norm
    elif norm_name == "A-wI":
        norm = A_wI_norm
    else:
        raise RuntimeError()

    beta = float(np.linalg.norm(xi))
    # xi = xi / beta
    # beta = 1

    (v, V, H, m) = arnoldi(t ** 2 * omega2_0, xi.flatten() / beta, exact_n, trunc=1)
    exact = get_lanczos_approx(V, H, beta, func_sparse_sym)
    exact = exact.flatten()

    if apply_err0:
        exact_norm = norm(exact)
    else:
        exact_norm = 1
    print(f"Exact norm: {norm(exact)}")
    plot_store["exact norm"] = exact_norm
    fig, ax = get_fig_ax()
    colors = Colors()
    idx, lanczos_errors = get_lanczos_errors(V, scipy.sparse.csc_array(H), beta, exact, func_sparse_sym, 2,
                                             upper=100, norm=norm)
    plot_store["Lanczos idx"] = idx
    plot_store["Lanczos error_norms"] = lanczos_errors
    lanc_plot = ax.plot(idx, lanczos_errors, color='black', label=r"m=$\infty$")
    markers = ["o", "^", "x"]
    j = 0
    for j, krylov_size in enumerate([20, 10, 6]):
        num_starts = bound_n // krylov_size + 1

        # Calculate the matrix exponential
        npfs, npupdate_norms, final_size = lanczos_method(t ** 2 * omega2_0,
                                                          xi.flatten(),
                                                          func_sparse,
                                                          krylov_size=krylov_size, max_starts=num_starts,
                                                          stopping_acc=-np.inf, arnoldi_acc=-np.inf,
                                                          stopping_decay=-np.inf)
        error_norms = [exact_norm] + list(norm(exact.reshape((-1, 1)) - npfs, axis=0))
        idx = get_index(final_size, krylov_size)
        name = f"m:{krylov_size}"
        plot_store[name + " errors"] = error_norms
        plot_store[name + " update_norms"] = npupdate_norms
        plot_store[name + " idx"] = idx
        line, = ax.plot(idx, error_norms, linestyle="solid", c=colors[j])
        line, = ax.plot(idx[1:], npupdate_norms, linestyle="none", c=colors[j], marker=markers[j], label=name)

    j += 1
    # Applied to exp(adiag(-Omega^2, I)), which has the union of the two spectra due to similarity
    ms, bounds = hochbruck_lubich(-max(evals), t, n=bound_n)
    bounds *= exact_norm
    name = f"HL"
    plot_store[name + " bounds"] = bounds
    plot_store[name + " ms"] = ms
    ax.plot(ms, bounds, label="HL", linestyle="-", c=colors[j])

    j += 1
    ms, bounds = chen_musco(min(evals), max(evals), w=w, n=bound_n, f=func_scalar, center=center, radius=radius)
    name = f"CGMM prio"
    plot_store[name + " bounds"] = bounds
    plot_store[name + " ms"] = ms
    ax.plot(ms, exact_norm * bounds, label="CGMM", linestyle="-", c=colors[j])

    j += 1
    HH, v = restarted_lanczos(t ** 2 * omega2_0, xi.flatten() / beta, krylov_size=krylov_size,
                              max_starts=bound_n // krylov_size + 1)
    ms, bounds = afanasjew_post_for_plot(HH.todense(), v, t ** 2 * omega2_0, krylov_size, starts=bound_n // krylov_size,
                                         f=func_dense)
    bounds *= beta * exact_norm
    name = f"Afanasjew 1 {krylov_size}"
    plot_store[name + " bounds"] = bounds[0, :]
    plot_store[name + " ms"] = ms
    ax.plot(ms, bounds[0, :], label="AEEG 1", linestyle="--", c=colors[j])
    name = f"Afanasjew 2 {krylov_size}"
    plot_store[name + " bounds"] = bounds[1, :]
    plot_store[name + " ms"] = ms
    ax.plot(ms, bounds[1, :], label="AEEG 2", linestyle="--", c=colors.get(j, True))

    np.savez(os.path.join(root_path, "artifacts", "plot_store" + f"protein_sinc_sqrt_{norm_name}-norm"), **plot_store)

    fig.suptitle(
        rf"${func_scalar.__name__}(h^2\Omega^2)b$" + r"$A\in\mathbb{R}^{" + rf"{N}\times{N}" + "}$")
    ax.set_yscale("log")
    ax.set_ylim(bottom=np.finfo(evals[0].dtype).eps / 1000)
    ax.set_xlabel("Lanczos iterations")
    ax.set_ylabel(r"Error $||\cdot||_{" + norm_name + r"}$")
    ax.legend(framealpha=.5, scatterpoints=1, numpoints=1)
    postprocess_style()
    fig.tight_layout()
    fig.savefig(os.path.join(root_path, f"figures/protein_sinc_sqrt_{norm_name}-norm.png"))
    plt.show()
