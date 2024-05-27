
from np_funm import funm_krylov_v2 as np_funm_krylov
from jax_funm import funm_krylov as jax_funm_krylov
from jax_funm import funm_krylov_jittable as jax_funm_krylov_jittable
from jax.experimental.sparse import BCSR, BCOO
from jax import numpy as jnp
import numpy as np
import scipy
import time
from get_3d_laplacian import get_3d_laplacian



if __name__ == "__main__":
    n = 20  # Interior grid points in each direction
    N = n ** 3  # Total number of interior points
    A = get_3d_laplacian(n, n, n)
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    z = np.linspace(0, 1, n)
    index = np.linspace(1, n, n)
    xv, yv, zv = np.meshgrid(x, y, z)
    xi, yi, zi = np.meshgrid(index, index, index)
    h = 1 / (n + 1)
    u0 = np.zeros_like(xv)# mesh width
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            for k in range(1, n + 1):
                sini = np.sin(i * xi * np.pi * h) + np.sin(j * yi * np.pi * h) + np.sin(k * zi * np.pi * h)
                u0[i - 1, j - 1, k - 1] = np.sum(sini) / (i + j + k)

    t = 0.1
    k = 1
    dt = t / k
    m = 30  # tbdetermined somehow
    param = {
        "restart_length": m,
        "num_restarts": k
    }

    start = time.time()
    # Calculate the matrix exponential
    npfs, npeigvals, npupdate_norms = np_funm_krylov(dt * A, u0.flatten(), param)
    #npnorms = list(np.linalg.norm(exact - npfs, axis=0))
    #plt.plot(restart_length * np.arange(1, param["num_restarts"] + 1), npnorms, label=f"m:{restart_length}")
    print(f"It took: {time.time() - start}")
    start = time.time()
    _ = jax_funm_krylov(dt * A.toarray(), jnp.array(u0.flatten()), param)
    print(f"Jax took: {time.time() - start}")
    start = time.time()
    _ = jax_funm_krylov_jittable(dt * A.toarray(), jnp.array(u0.flatten()), param)
    print(f"Jax jittable took: {time.time() - start}")
    # ABCSR = BCSR.fromdense(A.toarray())
    # start = time.time()
    # _ = jax_funm_krylov_jittable(dt * ABCSR, jnp.array(u0.flatten()), param)
    # print(f"Jax jittable sparse BCSR took: {time.time() - start}")
    ABCOO = BCOO.fromdense(A.toarray())
    start = time.time()
    _ = jax_funm_krylov_jittable(dt * ABCOO, jnp.array(u0.flatten()), param)
    print(f"Jax jittable sparse BCCO took: {time.time() - start}")
