import matplotlib.pyplot as plt
import numpy as np
from experiments.utils import get_fig_axs, get_fig_ax, Colors, postprocess_style, setup_latex

if __name__ == '__main__':
    setup_latex()
    fig, ax = get_fig_ax()
    colors = Colors()
    plot_storeA = np.load("artifacts/plot_store_bounds_exp_heat_25_A-norm.npz")
    plot_store2 = np.load("artifacts/plot_store_bounds_exp_heat_25_2-norm.npz")

    ax.plot(plot_store2["Lanczos" + " ids"], plot_store2["Lanczos" + " errors"], color='black')
    # lanc_plot = ax.plot(plot_storeA["Lanczos" + " ids"], plot_storeA["Lanczos" + " errors"], color='black')
    exact_normA = plot_storeA["exact norm"]
    lin_error_0 = plot_storeA["lin_error_0"]
    evals = plot_store2["evals"]
    i = 0
    ax.plot(plot_store2["HL" + " ms"], plot_store2["HL" + " bounds"], label="HL", linestyle="-", c=colors[i])

    i += 1
    name = f"CGMM prio"
    ax.plot(plot_storeA["CGMM prio" + " ms"],
            lin_error_0 * np.sqrt(max(np.abs(evals))) * plot_storeA["CGMM prio" + " bounds"], label="CGMM",
            linestyle="-", c=colors[i])

    ax.plot(plot_store2["CGMM prio nk" + " ms"], plot_store2["CGMM prio nk" + " bounds"], linestyle=":",
            c=colors.get(i, False))

    i += 1
    ax.plot(plot_store2["CGMM post nk" + " ms"], plot_store2["CGMM post nk" + " bounds"], linestyle=":",
            c=colors.get(i, False))
    ax.plot(plot_storeA["CGMM post" + " ms"],
            np.sqrt(max(np.abs(evals))) * lin_error_0 * plot_storeA["CGMM post" + " bounds"], label="CGMM",
            linestyle="--", c=colors[i])

    i += 1
    ax.plot(plot_store2["Saad post" + " ms"], plot_store2["Saad post" + " bounds"], label="S", linestyle="--",
            c=colors[i])

    ax.set_yscale("log")
    ax.set_ylim(bottom=np.finfo(np.float64).eps / 1000, top=10000)
    ax.set_xlabel("Lanczos iterations")
    # ax.set_ylabel(fr"Error $||\cdot||_A$")
    ax.legend(framealpha=.5, scatterpoints=1, numpoints=1)
    postprocess_style()
    fig.tight_layout()
    plt.savefig("figures/bounds_exp_heat_25_2-norm")
    fig.show()
