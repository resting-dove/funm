from src.matfuncb.np_funm import funm_krylov as np_funm_krylov
from src_jax.quad_expm_jax import expm_quad as expm_quad_jax
from scipy_expm import expm
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Set up a diagonal matrix
    n = 100
    EWs = -np.arange(1, n + 1) / n
    EWs[0:4] = [-200, -21, -20.99, 0]
    S = np.eye(n, n)

    A = S.transpose() @ np.diag(EWs) @ S
    b = np.ones(n) / np.linalg.norm(np.ones(n))

    param = {
        "restart_length": 7,
        "num_restarts": 10
    }

    # Calculate the matrix exponential
    exact = S.transpose() @ np.diag(np.exp(EWs)) @ S @ b

    #f, out = expm_quad(A, b, param["num_restarts"], param["restart_length"], keep_H=True, keep_V=True)
    f, out = expm_quad_jax(jnp.array(A), jnp.array(b), param["num_restarts"], param["restart_length"], keep_f=True)
    npfs, npeigvals, npupdate_norms = np_funm_krylov(A, b, param)
    scf, sceigvals = expm(A, b)

    #qnorms = [np.linalg.norm(exact - 0)] + list(np.linalg.norm(exact - f))
    norms = [np.linalg.norm(exact - 0)] + list(np.linalg.norm(exact.reshape((n, 1)) - np.array(out["f"]).T, axis=0))
    npnorms = [np.linalg.norm(exact - 0)] + list(np.linalg.norm(exact.reshape((n, 1)) - npfs, axis=0))
    scnorm = np.linalg.norm(exact - scf)

    print(scnorm)
    #plt.plot(np.arange(len(qnorms)), qnorms, label="quad")
    plt.plot(np.arange(len(norms)), norms, label="jax")
    plt.plot(np.arange(param["num_restarts"] + 1), npnorms, label="numpy")
    #plt.scatter(np.arange(1, param["num_restarts"] + 1), update_norms, label="jax update norms")
    plt.scatter(np.arange(1, len(out["update"]) + 1), out["update"], label="quad update norms")
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
        plt.scatter(np.real(npeigvals[k]), np.imag(npeigvals[k]), marker="x", label=k, s=20 - k)

    plt.legend()
    plt.ylabel("Imaginary")
    plt.xlabel("Real")
    plt.show()
