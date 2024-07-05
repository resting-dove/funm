from src_jax.jax_funm import funm_krylov_jittable
from src.matfuncb.np_funm import funm_krylov as np_funm_krylov
from scipy_expm import expm
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
import scipy

def get_block(j:int):
    return np.array([[0, 1],
                     [-1, 0]]) * j / 25
def get_skew_symmetric_matrix(j=500):
    return scipy.linalg.block_diag(0, *[get_block(j) for j in range(1, j + 1)])

if __name__ == "__main__":
    n = 1001
    A = get_skew_symmetric_matrix(int((n-1)/2))
    b = np.ones(n) / np.linalg.norm(np.ones(n))


    exact, _, _ = np_funm_krylov(A, b, {"restart_length": n, "num_restarts": 1})

    param = {
        "restart_length": 15,
        "num_restarts": 10
    }

    # Calculate the matrix exponential
    start = time.time()
    #fs, _ = jax_funm_krylov(A, b, param)
    print(f"Regular Jax took: {time.time() - start}")
    start = time.time()
    jfs, update_norms = funm_krylov_jittable(jnp.array(A), jnp.array(b), param)
    # eigvals = np.linalg.eigvals(H_full[:(k + 1) * m, :(k + 1) * m])
    print(f"Jitted Jax took: {time.time() - start}")
    start = time.time()
    npfs, npeigvals, npupdate_norms = np_funm_krylov(A, b, param)
    print(f"Numpy took: {time.time() - start}")
    start = time.time()
    scf, sceigvals = expm(A, b)
    print(f"Scipy Expm took: {time.time() - start}")


    norms = [np.linalg.norm(exact - 0)] + list(np.linalg.norm(exact.reshape((n, 1)) - fs, axis=0))
    jnorms = [np.linalg.norm(exact - 0)] + list(np.linalg.norm(exact.reshape((n, 1)) - jfs, axis=0))
    npnorms = [np.linalg.norm(exact - 0)] + list(np.linalg.norm(exact.reshape((n, 1)) - npfs, axis=0))
    scnorm = np.linalg.norm(exact - scf)

    print(scnorm)

    plt.plot(np.arange(param["num_restarts"] + 1), norms, label="jax")
    plt.plot(np.arange(param["num_restarts"] + 1), jnorms, label="jit")
    #plt.plot(np.arange(param["num_restarts"] + 1), npnorms, label="numpy")
    plt.scatter(np.arange(1, param["num_restarts"] + 1), update_norms, label="jax update norms")
    plt.scatter(np.arange(1, param["num_restarts"] + 1), npupdate_norms, label="numpy update norms")
    plt.title("Error of restarted expm for diagonal matrix")
    plt.legend(framealpha=.5)
    plt.yscale("log")
    plt.ylim(bottom=max(np.finfo(npfs.dtype).eps, plt.ylim()[0]) / 10)
    plt.xlabel("Restarts")
    plt.show()

    plt.title("Ritz values")
    for k in [0, param["num_restarts"] - 1]:
        plt.scatter(np.real(npeigvals[k]), np.imag(npeigvals[k]), marker="x", label=k, s=20 - k)

    plt.legend()
    plt.ylabel("Imaginary")
    plt.xlabel("Real")
    plt.show()
