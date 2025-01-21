import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import openmm.unit as unit
import scipy

from experiments.utils import get_fig_ax, Colors, postprocess_style, setup_latex, get_fig_axs
from gautschiIntegrators.gautschiIntegrators.lanczos.LanczosEvaluator import LanczosWkmEvaluator, \
    RestartedLanczosWkmEvaluator
from pywkm.wkm import wkm
from src.matfuncb.error_bounds import hochbruck_lubich, chen_musco, afanasjew_post_for_plot, restarted_post_no_kappa

root_path = os.getcwd()


def compute_error(x, x_true, rtol, atol, norm=scipy.linalg.norm):
    return norm(x - x_true) / norm(atol + rtol * x_true)

def cos_sqrt(x):
    return np.real(np.cos(np.sqrt(x + 0j)))


def cos_sqrtm(H):
    return wkm(-H, return_sinhc=False)


def sinc_sqrt(x):
    return np.real(np.sinc(np.sqrt(x + 0j) / np.pi))


if __name__ == '__main__':
    plot_store = {}
    plot_store["filename"] = os.path.basename(__file__)
    plot_store["git_commit"] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    setup_latex()
    fig, ax = get_fig_ax(factor=1)
    fig2, ax2 = get_fig_ax(factor=1)
    axs = [ax, ax2]
    colors = Colors()

    A = scipy.sparse.load_npz("OpenMM_Omega2_0.0femtosecond.npz")
    b = np.load("OpenMM_vectors_0.0femtosecond.npz")["xi"]
    N = A.shape[0]

    time_step = 10 * unit.femtosecond
    t = time_step.value_in_unit_system(unit.md_unit_system)

    evals = scipy.sparse.linalg.eigsh(A, return_eigenvectors=False, which="BE", k=5)
    # evals = np.array([-3857.46464699,  -3856.94705018, 488252.58628795, 488258.8403282, 488319.84228114])
    print("evals: ", evals)
    print(f"t={t}")
    plot_store["evals"] = evals
    plot_store["t"] = t

    func_dense = cos_sqrtm
    # func_sparse = sinc_sqrtm
    # func_sparse_sym = sinc_sqrtm
    func_scalar = cos_sqrt
    center, w = t**2* max(evals), min(-0.1, t ** 2 * 2 * min(evals))  # t ** 2 * min(evals), min(-0.1, t ** 2 * 2 * min(evals))  #
    radius = np.abs(center - w) * 2  # max(evals) * t ** 2 - w  #
    bound_n = 40
    norm_name = "2"
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


    exact_norm = norm(exactcA)
    print(f"Exact norm: {norm(exactcA)}")
    plot_store["exact norm"] = exact_norm
    lin_error_0 = A_wI_norm(scipy.sparse.linalg.spsolve(t ** 2 * A - w * scipy.sparse.eye(*A.shape), b))
    print(f"Linear error 0: {lin_error_0}")
    plot_store["lin_error_0"] = lin_error_0

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

    j = 0
    # Applied to exp(adiag(-Omega^2, I)), which has the union of the two spectra due to similarity
    ms, bounds = hochbruck_lubich(-max(evals), t**2, n=bound_n)
    bounds = bounds / exact_norm  # to make it relative
    name = f"HL"
    plot_store[name + " bounds"] = bounds
    plot_store[name + " ms"] = ms
    axs[0].plot(ms, bounds, label="HL", linestyle="-", c=colors[j])

    j += 1
    ms, bounds = chen_musco(t ** 2 * min(evals), t ** 2 * max(evals), w=w, n=bound_n, f=func_scalar, center=center,
                            radius=np.abs(center - w))
    bounds = bounds * lin_error_0 / exact_norm
    name = f"CGMM prio"
    plot_store[name + " bounds"] = bounds
    plot_store[name + " ms"] = ms
    axs[0].plot(ms, np.sqrt(t**2 * max(np.abs(evals))) * bounds, label="CGMM", linestyle="-", c=colors[j])
    j += 1

    prev = 0
    for k in [2, 4, 8]:  # np.arange(1, 50, 10):
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
        axs[1].plot([0] + list(rks), [1] + rerrorcA, color=colors.get(j), label=fr"$m={k}$")
        axs[1].scatter(rks, rupdatecA, color=colors.get(j))
        if k in [2, 8]:
            axs[0].plot([0] + list(rks), [1] + rerrorcA, color=colors.get(j), label=fr"$m={k}$")
            axs[0].scatter(rks, rupdatecA, color=colors.get(j))

        ms, bounds = restarted_post_no_kappa(t ** 2 * A, b, rlanczos.rlanczos.T, w=w, center=center,
                                             radius=radius,
                                             m=k, f=func_scalar, norm=norm)
        bounds = bounds / exact_norm * np.abs(center - w) / radius  # to make it relative and account for radius
        name = f"rest post nk {k}"
        plot_store[name + " bounds"] = bounds
        plot_store[name + " ms"] = ms
        axs[1].plot(ms, bounds, linestyle=":", c=colors.get(j, False))
        if k in [2, 8]:
            axs[0].plot(ms, bounds, linestyle=":", c=colors.get(j, False))

        # j += 1
        ms, bounds = afanasjew_post_for_plot(rlanczos.rlanczos.T, rlanczos.rlanczos.v_next, t ** 2 * A, k,
                                             starts=r,
                                             f=func_dense)
        bounds = bounds * beta / exact_norm  # to make it relative
        name = f"Afanasjew 1 {k}"
        plot_store[name + " bounds"] = bounds[0, :]
        plot_store[name + " ms"] = ms
        # axs[0].plot(ms, bounds[0, :], label="AEEG 1", linestyle="--", c=colors[j])
        name = f"Afanasjew 2 {k}"
        plot_store[name + " bounds"] = bounds[1, :]
        plot_store[name + " ms"] = ms
        axs[1].plot(ms, bounds[1, :], linestyle="--", c=colors.get(j, False))
        if k in [2, 8]:
            axs[0].plot(ms, bounds[1, :], linestyle="--", c=colors.get(j, False))
        j += 1

    name = f"m:infty"
    plot_store[name + " errorsA"] = errorsA
    plot_store[name + " errorcA"] = errorcA
    plot_store[name + " idx"] = ks
    line, = axs[0].plot([0] + list(ks), [1] + errorcA, c="black", label=r"$m=\infty$")
    axs[1].plot([0] + list(ks), [1] + errorcA, c="black", label=r"$m=\infty$")

    axs[0].set_yscale("log")
    axs[0].set_ylim(bottom=np.finfo(evals[0].dtype).eps / 100, top=5000000)
    axs[0].set_xlabel("Lanczos iterations")
    # axs[0].set_ylabel(r"rel. error $||\cdot||_{" + norm_name + "}$")
    axs[0].legend(framealpha=.5, scatterpoints=1, numpoints=1)
    axs[1].set_yscale("log")
    axs[1].set_ylim(bottom=np.finfo(evals[0].dtype).eps / 100, top=5000000)
    axs[1].set_xlabel("Lanczos iterations")
    # axs[0].set_ylabel(r"rel. error $||\cdot||_{" + norm_name + "}$")
    axs[1].legend(framealpha=.5, scatterpoints=1, numpoints=1)
    np.savez(
        os.path.join(root_path, "artifacts", "plot_store" + f"_bounds_cos_sqrt_{norm_name}-norm_{time_step}"),
        **plot_store)

    # fig.suptitle(
    #     rf"${func_scalar.__name__}(A)b$, $\Lambda(A)\subset[{min(evals):.1f}, {max(evals):.1f}]$, " + r"$A\in\mathbb{R}^{" + rf"{N}\times{N}" + "}$")
    postprocess_style()
    fig.tight_layout()
    fig.savefig(os.path.join(root_path, f"figures/bounds_cos_sqrt_{norm_name}-norm_{time_step}"))
    fig2.tight_layout()
    fig2.savefig(os.path.join(root_path, f"figures/restarted_bounds_cos_sqrt_{norm_name}-norm_{time_step}"))
    plt.show()
