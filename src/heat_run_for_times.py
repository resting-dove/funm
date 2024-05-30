import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
import json
from get_3d_laplacian import get_3d_laplacian, get_smallest_evs
from Arnoldi import arnoldi

root_path = os.getcwd()


def funm_krylov(A, b: np.array, param, exact, tol=1e-10):
    """Variation on funm_krylov_v2. Exact gives the correct solution and restarts are run to a max number or tolerance
    is reached."""
    n = b.shape[0]
    beta = float(np.linalg.norm(b))
    w = b / beta
    m = param["restart_length"]
    V_big = np.zeros((n, 0), b.dtype)
    f = np.zeros_like(b)
    H_full = np.zeros((m * param["max_num_restarts"] + 2, m * param["max_num_restarts"]), dtype=b.dtype)
    update_norms = []
    norms = [np.linalg.norm(exact - f)]
    k = 0
    while k < param["max_num_restarts"] and norms[-1] > tol:
        (w, new_V_big, H) = arnoldi(A=A, w=w, m=m)
        V_big = np.concatenate([V_big, new_V_big], axis=1)
        H_full[k * m: (k + 1) * m + 1, k * m: (k + 1) * m] = H
        H_exp = scipy.linalg.expm(H_full[: (k + 1) * m, : (k + 1) * m])
        H_exp_jax = np.array(H_exp)[-m:, 0]
        f = beta * (V_big[:, k * m: (k + 1) * m] @ H_exp_jax) + f
        norms.append(np.linalg.norm(exact - f))
        #update_norms.append(np.linalg.norm(beta * (V_big[:, k * m: (k + 1) * m] @ H_exp_jax)))
        k += 1
    return k, norms

if __name__ == "__main__":
    n = 20  # Interior grid points in each direction
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
    data_for_table = {}
    for restart_length in [7, 8, 11, 13, 22, 29, 36, 51]:
        num_restarts = 150 // restart_length + 1
        param = {
            "restart_length": restart_length,
            "max_num_restarts": num_restarts
        }

        # Calculate the matrix exponential
        start = time.time()
        k, npnorms = funm_krylov(t * A.toarray(), u0.flatten(), param, exact)
        duration = time.time() - start
        data_for_table[restart_length] = {"m": restart_length, "k": k, "time": duration, "error": npnorms[-1]}
        plt.plot(restart_length * np.arange(0, k + 1), npnorms, label=f"m:{restart_length}")
        #plt.scatter(np.arange(1, param["num_restarts"] + 1), npupdate_norms, label=f"m:{restart_length}")

    plt.title(f"Error for heat equation with n={n}")
    plt.legend(framealpha=.5)
    plt.yscale("log")
    plt.ylim(bottom=max(np.finfo(evals[0].dtype).eps, plt.ylim()[0]) / 10)
    plt.xlabel("Arnoldi iterations")
    plt.show()
    with open(os.path.join(root_path, f"tables/heat{n}times.json"), "w") as file:
        file.write(json.dumps(data_for_table))

    for k in data_for_table.keys():
         print(data_for_table.get(k))

