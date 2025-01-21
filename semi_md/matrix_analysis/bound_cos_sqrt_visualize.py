import matplotlib.pyplot as plt
import numpy as np
from experiments.utils import get_fig_axs, get_fig_ax, Colors, postprocess_style, setup_latex

if __name__ == '__main__':
    setup_latex()
    fig, ax = get_fig_ax()
    colors = Colors()
    markers = ["o", "^"]
    plot_store = np.load("artifacts/plot_store_bounds_cos_sqrt_2-norm_150 fs.npz")
    exact_norm = plot_store["exact norm"]
    evals = plot_store["evals"]
    t = plot_store["t"]
    j = 2
    for k in [2, 4, 8]:
        name = f"m:{k}"
        rerrorcA = plot_store[name + " errorcA"]
        rks = plot_store[name + " idx"]
        rupdatecA = plot_store[name + " updatecA"]

        ax.plot([0] + list(rks), [1] + list(rerrorcA), color=colors.get(j), label=fr"$m={k}$")
        # ax.scatter(rks, rupdatecA, color=colors.get(j))

        name = f"rest post nk {k}"
        bounds = plot_store[name + " bounds"]
        ms = plot_store[name + " ms"]
        ax.plot(ms, bounds, linestyle=":", c=colors.get(j, False))

        name = f"Afanasjew 2 {k}"
        bounds = plot_store[name + " bounds"]
        ms = plot_store[name + " ms"]
        ax.plot(ms, bounds, linestyle="--", c=colors.get(j, False))

        j += 1

    name = f"m:infty"
    errorcA = plot_store[name + " errorcA"]
    ks = plot_store[name + " idx"]
    ax.plot([0] + list(ks), [1] + list(errorcA), c="black", label=r"$m=\infty$")


    ax.set_yscale("log")
    ax.set_ylim(bottom=np.finfo(np.float64).eps / 100, top=10000)
    ax.set_xlabel("Lanczos iterations")
    ax.legend(framealpha=.5, scatterpoints=1, numpoints=1)

    postprocess_style()
    fig.tight_layout()
    fig.savefig("figures/restarted_bounds_cos_sqrt_2-norm_150 fs")
    fig.show()