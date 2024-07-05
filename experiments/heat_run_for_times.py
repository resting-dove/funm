import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
import json
from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.experimental.sparse
from jax import numpy as jnp
from src_jax.jax_Arnoldi import arnoldi_jittable
from src.matfuncb.Arnoldi import arnoldi
from functools import partial

root_path = os.getcwd()



@partial(jax.jit, static_argnames=["m", ])
def calculate_loop_p1(A, w, m, H_full, k):
    (w, V, H, breakdown) = arnoldi_jittable(A=A, w=w, m=m)
    #V_big = jnp.concat([V_big, new_V_big], axis=1)
    H_full = jax.lax.dynamic_update_slice(H_full, H, (k * m, k * m))
    return V, w, H_full, breakdown


@partial(jax.jit, static_argnames=["beta"])
def calculate_loop_p2(H_exp, beta, V, f):
    f = beta * (V @ H_exp) + f
    return f


# @partial(jax.jit, static_argnames=["m", "beta"], device=jax.devices('cpu')[0])
def calculate_loop(A: jax.Array, H_full: jax.Array, f: jax.Array, w: jax.Array, m: int, k: int,
                   beta: float):
    V, w, H_full, breakdown = calculate_loop_p1(A, w, m, H_full, k)

    # H_exp = jax.scipy.linalg.expm(H_full[: (k + 1) * m, : (k + 1) * m])
    H_slice = jax.lax.dynamic_slice(H_full, (0, 0), ((k + 1) * m, (k + 1) * m))
    H_exp = jax.scipy.linalg.expm(H_slice)[-m:, 0]
    f = calculate_loop_p2(H_exp, beta, V, f)
    return f, H_full, w, breakdown


def funm_krylov_jittable(A, b: jax.Array, param, exact, tol=1e-10):
    cpu_device = jax.devices('cpu')[0]
    n = b.shape[0]
    beta = float(jnp.linalg.norm(b))
    w = b / beta
    m = param["restart_length"]
    f = jnp.zeros_like(b)
    H_full = jnp.zeros((m * param["max_num_restarts"] + 1, m * param["max_num_restarts"]))
    norms = [jnp.linalg.norm(exact - f)]
    k = 0
    breakdown = False
    while k < param["max_num_restarts"] and norms[-1] > tol and breakdown is False:
        f, H_full, w, breakdown = calculate_loop(A, H_full, f, w, m, k, beta)
        norms.append(jnp.linalg.norm(exact - f))
        k += 1

    return k, norms


def funm_krylov(A, b: np.array, param, exact, tol=1e-10):
    """Variation on funm_krylov_v2. Exact gives the correct solution and restarts are run to a max number or tolerance
    is reached."""
    beta = float(np.linalg.norm(b))
    w = b / beta
    m = param["restart_length"]
    f = np.zeros_like(b)
    H_full = np.zeros((m * param["max_num_restarts"] + 2, m * param["max_num_restarts"]), dtype=b.dtype)
    norms = [np.linalg.norm(exact - f)]
    k = 0
    while k < param["max_num_restarts"] and norms[-1] > tol:
        (w, V, H) = arnoldi(A=A, w=w, m=m)
        H_full[k * m: (k + 1) * m + 1, k * m: (k + 1) * m] = H
        H_exp = scipy.linalg.expm(H_full[: (k + 1) * m, : (k + 1) * m])
        H_exp_jax = np.array(H_exp)[-m:, 0]
        f = beta * (V @ H_exp_jax) + f
        norms.append(np.linalg.norm(exact - f))
        k += 1
    return k, norms

if __name__ == "__main__":
    n = 15  # Interior grid points in each direction
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
    for restart_length in [7]: #, 8, 11, 13, 22, 29, 36, 51]:
        num_restarts = 150 // restart_length + 1
        param = {
            "restart_length": restart_length,
            "max_num_restarts": num_restarts
        }

        # Calculate the matrix exponential
        A_as_BCOO = jax.experimental.sparse.BCOO.from_scipy_sparse(A)
        start = time.time()
        k, npnorms = funm_krylov_jittable(t * A_as_BCOO, u0.flatten(), param, exact)
        duration = time.time() - start
        data_for_table[restart_length] = {"m": restart_length, "k": k, "time": duration, "error": npnorms[-1].tolist()}
        plt.plot(restart_length * np.arange(0, k + 1), npnorms, label=f"m:{restart_length}")
        #plt.scatter(np.arange(1, param["num_restarts"] + 1), npupdate_norms, label=f"m:{restart_length}")

    plt.title(f"Error for heat equation with n={n}")
    plt.legend(framealpha=.5)
    plt.yscale("log")
    plt.ylim(bottom=max(np.finfo(evals[0].dtype).eps, plt.ylim()[0]) / 10)
    plt.xlabel("Arnoldi iterations")
    plt.show()
    for k in data_for_table.keys():
        print(data_for_table.get(k))
    with open(os.path.join(root_path, f"tables/JaxSparseHeatJustV{n}times.json"), "w") as file:
         file.write(json.dumps(data_for_table))


