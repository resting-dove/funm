import os
import subprocess

import numpy as np
import scipy
import matplotlib.pyplot as plt
from gautschiIntegrators.gautschiIntegrators.lanczos.LanczosProvider import LanczosProvider
from gautschiIntegrators.gautschiIntegrators.lanczos.LanczosEvaluator import LanczosWkmEvaluator, \
    RestartedLanczosWkmEvaluator, \
    LanczosDiagonalizationEvaluator, RestartedLanczosDiagonalizationEvaluator
from pywkm.wkm import wkm
from experiments.utils import get_fig_ax, Colors, postprocess_style, get_fig_axs, setup_latex

root_path = os.getcwd()


def compute_error(x, x_true, rtol, atol, norm=scipy.linalg.norm):
    return norm(x - x_true) / norm(atol + rtol * x_true)


def cos_sqrt(x):
    return np.cos(np.sqrt(x))


def sinc_sqrt(x):
    return np.sinc(np.sqrt(x) / np.pi)


if __name__ == '__main__':
    plot_store = {}
    plot_store["filename"] = os.path.basename(__file__)
    plot_store["git_commit"] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    setup_latex()
    fig, axs = get_fig_axs(2, 1, sharex=True, factor=.6)
    colors = Colors()

    n = 200
    rng = np.random.default_rng()
    b = rng.random(n)
    b = b / scipy.linalg.norm(b)
    for idx, file in enumerate(
            ["randomSparse200_-2.000000.mat", "randomSparse200_-1.000000.mat", "randomSparse200_0.000000.mat",
             "randomSparse200_1.000000.mat", "randomSparse200_2.000000.mat", "randomSparse200_3.000000.mat"]):
        i = idx - 2
        mat_dict = scipy.io.loadmat("matlab_files/" + file)
        print(f"Sum of imaginary parts {np.sum(np.imag(mat_dict['A']))}")

        A = scipy.sparse.csr_array(np.real(mat_dict["A"]))
        print(
            f"Max eval should be of order 10^{i} and is {max(scipy.linalg.eigvalsh(A.todense()))}. The condition number is {mat_dict['conds']}.")

        fsA = mat_dict["fsA"]
        fcA = mat_dict["fcA"]
        condsA = mat_dict['conds']
        condcA = mat_dict['condc']
        exactsA = fsA @ b
        exactcA = fcA @ b

        ks = np.arange(1, 100, 8)
        errorsA = []
        rerrorsA = []
        errorcA = []
        rerrorcA = []
        for k in ks:
            # pass
            # lanczos = LanczosDiagonalizationEvaluator(krylov_size=k)
            lanczos = LanczosWkmEvaluator(krylov_size=k)
            lanczos.reset()
            cb, sb = lanczos.wave_kernels(1, A, b)
            errorsA.append(compute_error(sb, exactsA, atol=0, rtol=1))
            errorcA.append(compute_error(cb, exactcA, atol=0, rtol=1))
        rks = []
        for k in [10]:  # np.arange(1, 50, 10):
            rs = np.arange(100 // k)
            for r in rs:
                rks.append(r * k + k)
                # rlanczos = RestartedLanczosDiagonalizationEvaluator(krylov_size=k, max_restarts=r)
                rlanczos = RestartedLanczosWkmEvaluator(krylov_size=k, max_restarts=r)
                cb, sb = rlanczos.wave_kernels(1, A, b)
                rerrorsA.append(compute_error(sb, exactsA, atol=0, rtol=1))
                rerrorcA.append(compute_error(cb, exactcA, atol=0, rtol=1))

        name = f"m:infty"
        plot_store[name + " errorsA"] = errorsA
        plot_store[name + " errorcA"] = errorcA
        plot_store[name + " idx"] = ks

        name = f"m:{k}"
        plot_store[name + " errorsA"] = rerrorsA
        plot_store[name + " errorcA"] = rerrorcA
        plot_store[name + " idx"] = rks
        line, = axs[0].plot(ks, errorsA, label=f"10^{i}", c=colors[idx])
        line, = axs[1].plot(ks, errorcA, label=f"10^{i}", c=colors[idx])
        axs[0].plot(rks, rerrorsA, linestyle=":", c=colors[idx])
        axs[1].plot(rks, rerrorcA, linestyle=":", c=colors[idx])

        axs[0].plot(rks[-1] + 10, mat_dict["condc"] * np.finfo(np.float64).eps, linestyle=None, c=colors.get(idx, False), marker="x")
        axs[1].plot(rks[-1] + 10, mat_dict["conds"] * np.finfo(np.float64).eps, linestyle=None, c=colors.get(idx, False), marker="x")

    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    # ax.set_ylim(bottom=np.finfo(evals[0].dtype).eps / 1000, top=10000)
    axs[1].set_xlabel("Lanczos iterations")
    # axs[0].set_ylabel(r"rel. error $||\cdot||_{2}$")
    # axs[1].set_ylabel(r"rel. error $||\cdot||_{2}$")
    axs[0].legend(framealpha=.5, scatterpoints=1, numpoints=1)
    np.savez(
        os.path.join(root_path, "artifacts", "plot_store" + f"_analyse_PyWkm_convergence"),
        **plot_store)

    # fig.suptitle(
    #     rf"${func_scalar.__name__}(A)b$, $\Lambda(A)\subset[{min(evals):.1f}, {max(evals):.1f}]$, " + r"$A\in\mathbb{R}^{" + rf"{N}\times{N}" + "}$")
    postprocess_style()
    fig.tight_layout()
    # plt.savefig(os.path.join(root_path, f"figures/bounds_exp_uniform_restarts_{n}.png"))
    fig.savefig(os.path.join(root_path, f"figures/analyse_PyWkm_convergence"))
