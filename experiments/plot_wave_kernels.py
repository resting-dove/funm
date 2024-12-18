import numpy as np
import matplotlib.pyplot as plt
from utils import get_fig_ax, Colors, postprocess_style
import os


def wkms(x):
    return np.real(np.sinc(np.sqrt(x + 0j) / np.pi))


def wkmc(x):
    return np.real(np.cos(np.sqrt(x + 0j)))


if __name__ == "__main__":
    fig, ax = get_fig_ax(factor=0.5)
    colors = Colors()
    xs = np.linspace(-10, 100, 5000)
    ax.plot(xs, wkmc(xs), color=colors[0], label="1")
    ax.plot(xs, wkms(xs), color=colors[1], label="2")
    fig.legend()
    postprocess_style()
    fig.savefig("figures/plot_wave_kernels.png")
    plt.show()
