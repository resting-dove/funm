import os
import subprocess

import numpy as np
import scipy
import matplotlib.pyplot as plt
import openmm.unit as unit
from semi_md.ase_md.unit_helpers import ase_unit_system
from gautschiIntegrators.gautschiIntegrators.lanczos.LanczosProvider import LanczosProvider
from gautschiIntegrators.gautschiIntegrators.lanczos.LanczosEvaluator import LanczosWkmEvaluator, \
    RestartedLanczosWkmEvaluator, \
    LanczosDiagonalizationEvaluator, RestartedLanczosDiagonalizationEvaluator
from pywkm.wkm import wkm
from experiments.utils import get_fig_ax, Colors, postprocess_style, get_fig_axs
from src.matfuncb.error_bounds import hochbruck_lubich, chen_musco, afanasjew_post_for_plot

root_path = os.getcwd()


def compute_error(x, x_true, rtol, atol, norm=scipy.linalg.norm):
    e = (x - x_true) / (atol + rtol * np.abs(x_true))
    return norm(e) / np.sqrt(e.shape[0])


def cos_sqrt(x):
    return np.cos(np.sqrt(x))


def cos_sqrtm(H):
    return wkm(-H, return_sinhc=False)


def sinc_sqrt(x):
    return np.sinc(np.sqrt(x) / np.pi)


if __name__ == '__main__':
    plot_store = {}
    plot_store["filename"] = os.path.basename(__file__)
    plot_store["git_commit"] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

    fig, ax = get_fig_ax()
    colors = Colors()

    A = scipy.sparse.load_npz("OpenMM_K.npz")  # scipy.sparse.load_npz("Omega2_0.0femtosecond.npz")
    Msqrt_inv = scipy.sparse.diags_array(
        1 / scipy.sparse.load_npz("proteinMsqrt.npz").diagonal())
    A = Msqrt_inv @ A @ Msqrt_inv
    b = np.load("vectors_0.0femtosecond.npz")["xi"]
    b = (b * unit.angstrom).value_in_unit_system(unit.md_unit_system)
    N = A.shape[0]

    time_step = 10 * unit.femtosecond
    # ase_time_step = time_step.value_in_unit(unit.femtosecond) * ase_units.fs
    # t = ase_time_step
    t = time_step.value_in_unit_system(unit.md_unit_system)  # time_step.value_in_unit_system(ase_unit_system)

    evals = scipy.sparse.linalg.eigsh(A, return_eigenvectors=False, which="BE", k=5)
    print("evals: ", evals)
    print(f"t={t}")
    plot_store["evals"] = evals
    plot_store["t"] = t

    func_dense = cos_sqrtm
    # func_sparse = sinc_sqrtm
    # func_sparse_sym = sinc_sqrtm
    func_scalar = cos_sqrt
    center, w = 0, min(-0.1, t ** 2 * 2 * min(evals))  # max(evals), min(-0.1, t ** 2 * 2 * min(evals))
    radius = max(evals) * t ** 2 - w  # np.abs(center - w)
    bound_n = 40
    norm_name = "A-wI"
    apply_err0 = False
    exact_n = 2000


    def A_norm(x, axis=None):
        if axis is None:
            return np.sqrt(x.T @ (t ** 2 * A) @ x)
        elif axis == 1:
            return [A_norm(x[i, :]) for i in range(x.shape[0])]
        elif axis == 0:
            return [A_norm(x[:, i]) for i in range(x.shape[1])]
        else:
            raise RuntimeError()


    def A_wI_norm(x, axis=None):
        if axis is None:
            return np.sqrt(x.T @ (t ** 2 * A - w * scipy.sparse.eye(*A.shape)) @ x)
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

    beta = float(np.linalg.norm(b))

    lanczos = LanczosWkmEvaluator(krylov_size=exact_n)
    lanczos.reset()
    # fcA, fsA = wkm(-t**2*A, return_sinhc=True)
    # exactcA, exactsA = fcA @ b, fsA @ b
    exactcA, exactsA = lanczos.wave_kernels(t, A, b)

    if apply_err0:
        exact_norm = norm(exactcA)
    else:
        exact_norm = 1
        print(f"Exact norm: {norm(exactcA)}")
    plot_store["exact norm"] = exact_norm

    ks = np.arange(1, bound_n)
    errorsA = []
    errorcA = []
    for k in ks:
        # pass
        # lanczos = LanczosDiagonalizationEvaluator(krylov_size=k)
        lanczos = LanczosWkmEvaluator(krylov_size=k)
        lanczos.reset()
        cb, sb = lanczos.wave_kernels(t, A, b)
        errorsA.append(compute_error(sb, exactsA, atol=0, rtol=1, norm=norm))
        errorcA.append(compute_error(cb, exactcA, atol=0, rtol=1, norm=norm))

    j = 1
    ms, bounds = chen_musco(t ** 2 * min(evals), t ** 2 * max(evals), w=w, n=bound_n, f=func_scalar, center=center,
                            radius=radius)
    name = f"CGMM prio"
    plot_store[name + " bounds"] = bounds
    plot_store[name + " ms"] = ms
    ax.plot(ms, exact_norm * bounds, label="CGMM", linestyle="-", c=colors[j])
    j += 1

    prev = 0
    for k in [2, 8]:  # np.arange(1, 50, 10):
        rs = np.arange(bound_n // k)
        rupdatecA = []
        rks = []
        rerrorsA = []
        rerrorcA = []
        for r in rs:
            rks.append(r * k + k)
            # rlanczos = RestartedLanczosDiagonalizationEvaluator(krylov_size=k, max_restarts=r)
            rlanczos = RestartedLanczosWkmEvaluator(krylov_size=k, max_restarts=r)
            cb, sb = rlanczos.wave_kernels(t, A, b)
            rerrorsA.append(compute_error(sb, exactsA, atol=0, rtol=1, norm=norm))
            rerrorcA.append(compute_error(cb, exactcA, atol=0, rtol=1, norm=norm))
            rupdatecA.append(scipy.linalg.norm(cb - prev))
            prev = cb
        name = f"m:{k}"
        plot_store[name + " errorsA"] = rerrorsA
        plot_store[name + " errorcA"] = rerrorcA
        plot_store[name + " idx"] = rks
        plot_store[name + " updatecA"] = rupdatecA
        ax.plot([0] + list(rks), [exact_norm] + rerrorcA, linestyle=":", color=colors.get(j), label=k)
        ax.scatter(rks, rupdatecA, color=colors.get(j))
        j += 1

        ms, bounds = afanasjew_post_for_plot(rlanczos.rlanczos.T, rlanczos.rlanczos.v_next, t ** 2 * A, k,
                                             starts=r,
                                             f=func_dense)
        bounds *= beta * exact_norm
        name = f"Afanasjew 1 {k}"
        plot_store[name + " bounds"] = bounds[0, :]
        plot_store[name + " ms"] = ms
        ax.plot(ms, bounds[0, :], label="AEEG 1", linestyle="--", c=colors[j])
        name = f"Afanasjew 2 {k}"
        plot_store[name + " bounds"] = bounds[1, :]
        plot_store[name + " ms"] = ms
        # ax.plot(ms, bounds[1, :], label="AEEG 2", linestyle="--", c=colors.get(j, True))
        j += 1

    name = f"m:infty"
    plot_store[name + " errorsA"] = errorsA
    plot_store[name + " errorcA"] = errorcA
    plot_store[name + " idx"] = ks
    line, = ax.plot([0] + list(ks), [exact_norm] + errorcA, c="black")

    ax.set_yscale("log")
    ax.set_ylim(bottom=np.finfo(evals[0].dtype).eps / 1000, top=500000)
    ax.set_xlabel("Lanczos iterations")
    ax.set_ylabel(r"rel. error $||\cdot||_{" + norm_name + "}$")
    ax.legend(framealpha=.5, scatterpoints=1, numpoints=1)
    np.savez(
        os.path.join(root_path, "artifacts", "plot_store" + f"_bounds_cos_sqrt_omm_{norm_name}-norm"),
        **plot_store)

    # fig.suptitle(
    #     rf"${func_scalar.__name__}(A)b$, $\Lambda(A)\subset[{min(evals):.1f}, {max(evals):.1f}]$, " + r"$A\in\mathbb{R}^{" + rf"{N}\times{N}" + "}$")
    postprocess_style()
    fig.tight_layout()
    # plt.savefig(os.path.join(root_path, f"figures/bounds_exp_uniform_restarts_{n}.png"))
    fig.savefig(os.path.join(root_path, f"figures/bounds_cos_sqrt_omm_{norm_name}-norm.png"))
    plt.show()
