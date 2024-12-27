import matplotlib.pyplot as plt
import numpy as np
from experiments.utils import get_fig_axs, get_fig_ax, Colors, postprocess_style, setup_latex

if __name__ == '__main__':
    setup_latex()
    fig, ax = get_fig_ax()
    colors = Colors()
    markers = ["o", "^"]
    plot_store = np.load("artifacts/plot_storelarge_heat50_A-norm.npz")

    markers = ["o", "^", "x"]
    for j, krylov_size in enumerate([20, 10, 6]):
        name = f"m:{krylov_size}"
        error_norms = plot_store[name + " errors"]
        npupdate_norms = plot_store[name + " update_norms"]
        idx = plot_store[name + " idx"]
        line, = ax.plot(idx, error_norms, linestyle="solid", c=colors[j])
        line, = ax.plot(idx[1:], npupdate_norms, linestyle="none", c=colors[j], marker=markers[j], label=name)

    ax.set_yscale("log")
    ax.set_ylim(bottom=np.finfo(np.float64).eps / 1000)
    ax.set_xlabel("Lanczos iterations")
    ax.set_ylabel(r"Error $||\cdot||_A$")
    ax.legend(framealpha=.5, scatterpoints=1, numpoints=1)
    postprocess_style()
    fig.tight_layout()
    fig.savefig("figures/large_heat_50_restarts_A-norm")
