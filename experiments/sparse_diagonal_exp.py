from src_jax.jax_funm import funm_krylov_jittable as jax_funm_krylov_v2
from src.matfuncb.np_funm import funm_krylov as np_funm_krylov
from src.matfuncb.np_funm import funm_krylov_v2 as np_funm_krylov_v2
from scipy_expm import expm
import numpy as np
import jax.numpy as jnp
import scipy
from jax.experimental.sparse import BCOO
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Set up a diagonal matrix
    n = 100
    EWs = -np.arange(1, n + 1) / n
    EWs[0:4] = [-200, -21, -20.99, 0]
    S = scipy.sparse.eye(n, n)

    A = S.transpose() @ scipy.sparse.diags_array(EWs) @ S
    b = np.ones(n) / np.linalg.norm(np.ones(n))

    param = {
        "restart_length": 5,
        "num_restarts": 20
    }

    # Calculate the matrix exponential
    fs = jax_funm_krylov_v2(BCOO.fromdense(A.toarray()), jnp.array(b), param)
    fs1, eigvals1, update_norms1 = np_funm_krylov_v2(A, b, param)
    fs2, eigvals2, update_norms2 = np_funm_krylov_v2(A.toarray(), b, param)
    npfs, npeigvals, npupdate_norms = np_funm_krylov(A.toarray(), b, param)
    scf, sceigvals = expm(A.toarray(), b)

    exact = S.transpose() @ np.diag(np.exp(EWs)) @ S @ b

    norms = [np.linalg.norm(exact - 0)] + list(np.linalg.norm(exact.reshape((n, 1)) - fs, axis=0))
    norms1 = [np.linalg.norm(exact - 0)] + list(np.linalg.norm(exact.reshape((n, 1)) - fs1, axis=0))
    npnorms = [np.linalg.norm(exact - 0)] + list(np.linalg.norm(exact.reshape((n, 1)) - npfs, axis=0))
    npnorms2 = [np.linalg.norm(exact - 0)] + list(np.linalg.norm(exact.reshape((n, 1)) - fs2, axis=0))
    scnorm = np.linalg.norm(exact - scf)

    print(scnorm)

    plt.plot(np.arange(param["num_restarts"] + 1), norms, label="jax sparse")
    plt.plot(np.arange(param["num_restarts"] + 1), norms1, label="np sparse")
    plt.plot(np.arange(param["num_restarts"] + 1), npnorms, label="numpy orig")
    plt.plot(np.arange(param["num_restarts"] + 1), npnorms2, label="np dense")
    plt.scatter(np.arange(1, param["num_restarts"] + 1), update_norms1, label="jax update norms")
    plt.scatter(np.arange(1, param["num_restarts"] + 1), npupdate_norms, label="numpy update norms")
    plt.title("Error of restarted expm for diagonal matrix")
    plt.legend(framealpha=.5)
    plt.yscale("log")
    plt.ylim(bottom=max(np.finfo(EWs[0].dtype).eps, plt.ylim()[0], scnorm) / 10)
    plt.xlabel("Restarts")
    plt.show()

    plt.title("Ritz values of restarted expm for diagonal matrix")
    plt.scatter(np.real(EWs), np.imag(EWs), marker="o")
    #plt.scatter(np.real(sceigvals), np.imag(sceigvals), marker="x", label="scipy", s=10)
    for k in [0, param["num_restarts"] - 1]:
        plt.scatter(np.real(eigvals1[k]), np.imag(eigvals1[k]), marker="x", label=k, s=20 - k)

    plt.legend()
    plt.ylabel("Imaginary")
    plt.xlabel("Real")
    plt.show()
