import matplotlib.pyplot as plt
import numpy as np
from experiments.utils import get_fig_axs, get_fig_ax, Colors, postprocess_style, setup_latex

if __name__ == '__main__':
    setup_latex()
    fig, axs = get_fig_axs(2, 1, sharex=True)
    colors = Colors()
    markers = ["o", "^"]
    plot_store = np.load("artifacts/plot_storebounds_exp_heat_25_restarts_A-norm.npz")

    line, = axs[0].plot(plot_store["m:5" + " idx"], plot_store["m:5" + " errors"], label="m:5", linestyle="solid",
                        c="black")
    line, = axs[0].plot(plot_store["m:5" + " idx"][1:], plot_store["m:5" + " updates"], linestyle="none", c="black",
                        marker="^")

    line, = axs[1].plot(plot_store["m:20" + " idx"], plot_store["m:20" + " errors"], label="m:20", linestyle="solid",
                        c="black")
    line, = axs[1].plot(plot_store["m:20" + " idx"][1:], plot_store["m:20" + " updates"], linestyle="none", c="black",
                        marker="^")

    i = 2
    axs[0].plot(plot_store["rest post nk 5" + " ms"], plot_store["rest post nk 5" + " bounds"], label="CGMM",
                linestyle=":", c=colors[i])
    axs[1].plot(plot_store["rest post nk 20" + " ms"], plot_store["rest post nk 20" + " bounds"], label="CGMM",
                linestyle=":", c=colors[i])

    i += 1
    axs[0].plot(plot_store["Afanasjew 1 5" + " ms"], plot_store["Afanasjew 1 5" + " bounds"], label="AEEG 1",
                linestyle="--", c=colors[i])
    axs[0].plot(plot_store["Afanasjew 2 5" + " ms"], plot_store["Afanasjew 2 5" + " bounds"], label="AEEG 2",
                linestyle="--", c=colors[i + 1])
    axs[1].plot(plot_store["Afanasjew 1 20" + " ms"], plot_store["Afanasjew 1 20" + " bounds"], label="AEEG 1",
                linestyle="--", c=colors[i])
    axs[1].plot(plot_store["Afanasjew 2 20" + " ms"], plot_store["Afanasjew 2 20" + " bounds"], label="AEEG 2",
                linestyle="--", c=colors[i + 1])

    axs[0].set_yscale("log")
    axs[0].set_ylim(bottom=np.finfo(np.float64).eps / 1000, top=10000)
    axs[-1].set_xlabel("Lanczos iterations")
    # axs[0].set_ylabel(fr"Error $||\cdot||_A$")
    axs[0].legend(framealpha=.5, scatterpoints=1, numpoints=1)
    axs[1].set_yscale("log")
    axs[1].set_ylim(bottom=np.finfo(np.float64).eps / 1000, top=10000)
    axs[-1].set_xlabel("Lanczos iterations")
    # axs[1].set_ylabel(fr"Error $||\cdot||_A$")
    axs[1].legend(framealpha=.5, scatterpoints=1, numpoints=1)

    postprocess_style()
    fig.tight_layout()
    fig.savefig("figures/bounds_exp_heat_25_restarts_A-norm")
