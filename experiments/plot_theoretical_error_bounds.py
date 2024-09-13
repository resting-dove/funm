import numpy as np
import matplotlib.pyplot as plt
import scipy


def chen(t, rho, m):
    kappa = (4 * rho + 1)
    integral = 4 * rho + 1 * np.exp(t)  #
    cg = 2 * ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** m  # 2 * np.exp(-2 * m / np.sqrt(4 * rho + 1))
    return integral * cg


def hochbruck(t, rho, ms):
    out = np.empty_like(ms) * np.nan
    out = np.where(ms >= 2 * rho * t, 10 / (rho * t) / np.exp(rho * t) * (np.exp(1) * rho * t / ms) ** ms, out)
    indices = (2 * rho * t >= ms) * (ms >= np.sqrt(4 * rho * t))
    out = np.where(indices, 10 / np.exp(ms ** 2 / (5 * rho * t)), out)
    return out


def saad(norm, m):
    return 2 * norm ** m * np.exp(norm) / scipy.special.factorial(m)


if __name__ == '__main__':
    rho = 10
    t = 1
    ms = np.arange(1, 101)
    print(chen(t, rho, ms))
    print(hochbruck(t, rho, ms))
    saad_out = saad(4 * t * rho, ms)
    print(saad_out)
    plt.plot(ms, chen(t, rho, ms), label="Chen", linestyle='--', color='black')
    plt.plot(ms, hochbruck(t, rho, ms), label="Hochbruck", linestyle='-.', color='b')
    if np.min(saad_out[saad_out > 0]) < 100:
        plt.plot(ms, saad_out, label="Saad", linestyle=':', color='k')
    plt.ylim(1e-20, 1000)
    plt.yscale("log")
    plt.legend()

    plt.savefig("bounds.png")
    plt.show()
