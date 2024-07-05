import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.matfuncb.np_funm import funm_krylov, funm_krylov_v2
import os

root_path = os.getcwd()

def get_B(n):
    h = 1 / (n + 1)
    return 1 / h**2 * scipy.sparse.diags_array([1, -2, 1], offsets=[-1, 0, 1], shape=(n, n))

def get_Cj(n, mu):
    h = 1 / (n + 1)
    return 1 / h**2 * scipy.sparse.diags_array([1 + mu, -2, 1 - mu], offsets=[-1, 0, 1], shape=(n, n))

if __name__ == "__main__":
    n = 15
    nx, ny, nz = n, n, n
    h = 1/ (n + 1)
    N = nx * ny * nz
    tau1 = 320  # 96
    tau2 = 320  # 128
    mu1 = tau1 * h / 2
    mu2 = tau2 * h / 2
    In = scipy.sparse.eye(n, n)
    A = scipy.sparse.kron(In, scipy.sparse.kron(In, get_Cj(n, mu1))) + \
        scipy.sparse.kron(scipy.sparse.kron(get_B(n), In) + scipy.sparse.kron(In, get_Cj(n, mu2)), In)
    t = h ** 2
    phi = 1 + np.sqrt(1 - mu1**2 + 0j) + np.sqrt(1 - mu2**2 + 0j)
    spectrum_tA_real = [-6 - 2 * np.cos(np.pi * h) * np.real(phi), -6 + 2 * np.cos(np.pi * h) * np.real(phi)]
    spectrum_tA_im = [-2 * np.cos(np.pi * h) * np.imag(phi), 2 * np.cos(np.pi * h) * np.imag(phi)]

    b = np.ones(N)
    exact, _, _ = funm_krylov_v2(t * A, b, {"restart_length": int(n ** 2), "num_restarts":1})
    for restart_length in [2, 4, 6]:
        num_restarts = 90 // restart_length + 1
        param = {
            "restart_length": restart_length,
            "num_restarts": num_restarts
        }

        # Calculate the matrix exponential
        npfs, npeigvals, npupdate_norms = funm_krylov(t * A.toarray(), b, param)
        #npfs = funm_krylov_jittable(jnp.array(t * A.toarray()), b, param)
        npnorms = list(np.linalg.norm(exact - npfs, axis=0))
        plt.plot(restart_length * np.arange(1, param["num_restarts"] + 1), npnorms, label=f"m:{restart_length}")
        #plt.scatter(np.arange(1, param["num_restarts"] + 1), npupdate_norms, label=f"m:{restart_length}")
    plt.title(f"Convection - Diffusion example with n={n} and \mu_1={mu1}, \mu_2={mu2}")
    plt.legend(framealpha=.5)
    plt.yscale("log")
    plt.ylim(bottom=max(np.finfo(npfs.dtype).eps, plt.ylim()[0]) / 10)
    plt.xlabel("Arnoldi iterations")
    plt.savefig(os.path.join(root_path, f"figures/convDiff{tau1}ErrorPlot.png"))
    plt.show()

    plt.title(f"Ritz values for Convection - Diffusion example with n={n} and \mu_1={mu1}, \mu_2={mu2}")
    ax = plt.gca()
    rect = patches.Rectangle((spectrum_tA_real[0], spectrum_tA_im[0]),
                             spectrum_tA_real[1] - spectrum_tA_real[0],
                             spectrum_tA_im[1] - spectrum_tA_im[0],
                             linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    # plt.scatter(np.real(sceigvals), np.imag(sceigvals), marker="x", label="scipy", s=10)
    for k in [param["num_restarts"] - 1]:
        plt.scatter(np.real(npeigvals[k]), np.imag(npeigvals[k]), marker="x", label=(k + 1) * restart_length, s=20 - k)

    plt.legend()
    plt.ylabel("Imaginary")
    plt.xlabel("Real")
    plt.savefig(os.path.join(root_path, f"figures/convDiff{tau1}RitzValues.png"))
    plt.show()
    1 + 1