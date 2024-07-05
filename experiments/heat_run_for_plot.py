import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
from src.matfuncb.np_funm import funm_krylov

root_path = os.getcwd()

if __name__ == "__main__":
    n = 25  # Interior grid points in each direction
    N = n ** 3  # Total number of interior points
    h = 1 / (n + 1)
    x = np.linspace(h, 1 - h, n)
    y = np.linspace(h, 1 - h, n)
    z = np.linspace(h, 1 - h, n)
    index = np.linspace(1, n, n)
    xv, yv, zv = np.meshgrid(x, y, z)
    xi, yi, zi = np.meshgrid(index, index, index)
    #A = -1 / h ** 2 * get_3d_laplacian(n, n, n)
    lap = scipy.sparse.linalg.LaplacianNd((n, n, n), boundary_conditions='dirichlet')
    A = 1 / h ** 2 * lap.tosparse()
    u0 = np.zeros_like(xv)  # mesh width
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            for k in range(1, n + 1):
                sini = np.sin(i * xi * np.pi * h) + np.sin(j * yi * np.pi * h) + np.sin(k * zi * np.pi * h)
                u0[i - 1, j - 1, k - 1] = np.sum(sini) / (i + j + k)

    t = 0.1
    evals = t / h ** 2 * lap.eigenvalues(N)
    evecs = lap.eigenvectors(N)
    exact = (evecs @ np.diagflat(np.exp(evals)) @ evecs.T @ u0.flatten())
    for restart_length in [20, 10, 6]:
        num_restarts = 500 // restart_length + 1
        param = {
            "restart_length": restart_length,
            "num_restarts": num_restarts
        }

        # Calculate the matrix exponential
        npfs, _, npupdate_norms = funm_krylov(t * A.toarray(), u0.flatten(), param)
        npnorms = [np.linalg.norm(exact - 0)] + list(np.linalg.norm(exact.reshape((-1, 1)) - npfs, axis=0))
        plt.plot(restart_length * np.arange(0, param["num_restarts"] + 1), npnorms, label=f"m:{restart_length}")
        #plt.scatter(np.arange(1, param["num_restarts"] + 1), npupdate_norms, label=f"m:{restart_length}")

    plt.title(f"Error for heat equation with n={n}")
    plt.legend(framealpha=.5)
    plt.yscale("log")
    plt.ylim(bottom=max(np.finfo(evals[0].dtype).eps, plt.ylim()[0]) / 10)
    plt.xlabel("Arnoldi iterations")
    plt.savefig(os.path.join(root_path, f"figures/heat{n}ErrorPlot.png"))
    plt.show()
    1 + 1
