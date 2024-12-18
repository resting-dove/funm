import matplotlib.pyplot as plt
import numpy as np
from experiments.utils import get_fig_axs, get_fig_ax, Colors, postprocess_style, setup_latex

if __name__ == '__main__':
    setup_latex()
    gs99 = np.load("OneStepGS99.npz")
    adaptive = np.load("OneStepGS99_adaptive.npz")
    fig, ax = get_fig_ax(ratio=3)
    colors = Colors()
    eH = np.abs(gs99["Hs"] - adaptive["Hs"]) / gs99["Hs"]
    ax.plot(np.linspace(0, 1000, len(gs99["Hs"])), eH, label=r"$H$", color=colors[0])
    eI11 = np.abs(gs99["I11"] - adaptive["I11"]) / gs99["I11"]
    ax.plot(np.linspace(0, 1000, len(gs99["Hs"])), eI11, label=r"$I_{1,1}$", color=colors[1])
    eI12 = np.abs(gs99["I12"] - adaptive["I12"]) / gs99["I12"]
    ax.plot(np.linspace(0, 1000, len(gs99["Hs"])), eI12, label=r"$I_{1,2}$", color=colors[2])
    ax.set_yscale("log")
    ax.set_ylim(bottom=np.finfo(np.float64).eps / 1000, top=1)
    ax.set_xlabel(r"t")
    ax.set_ylabel(r"rel. Error")
    fig.legend()
    postprocess_style()
    fig.tight_layout()
    plt.savefig(f"relativeErrorsOneStepGS99")

    long = np.load("OneStepGS99Longer.npz")
    fig, axs = get_fig_axs(rows=2, cols=1, ratio=3, sharex=True)
    ax = axs[0]
    ax2 = axs[1]
    colors = Colors()
    ax.plot(np.linspace(0, 20000, len(long["Hs"])), long["Hs"], color=colors.get(0))
    ax2.plot(np.linspace(0, 20000, len(long["Hs"])), long["I11"], label=r"$I_{1,1}$", color=colors.get(1))
    ax2.plot(np.linspace(0, 20000, len(long["Hs"])), long["I12"], label=r"$I_{1,2}$", color=colors.get(2))

    ax.set_ylabel(r"H")
    ax2.set_ylabel(r"I")
    ax2.set_xlabel(r"t")
    ax.set_ylim(bottom=0)
    # ax2.set_ylim(bottom=min(gs99["I11"]), top=min(gs99["I11"]) + max(gs99["Hs"]) - min(gs99["Hs"]))
    ax2.legend()
    postprocess_style()
    fig.tight_layout()
    plt.savefig(f"EnergyConservationOneStepGS99Long")
