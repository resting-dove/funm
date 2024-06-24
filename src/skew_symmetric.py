
import scipy
from np_funm import funm_krylov_v2 as np_funm_krylov
import numpy as np
import matplotlib.pyplot as plt
import os
import time

root_path = os.getcwd()


def get_block(j: int):
    return np.array([[0, 1],
                     [-1, 0]]) * j / 25


def get_skew_symmetric_matrix(j=500):
    return scipy.sparse.block_diag((0, *[get_block(j) for j in range(1, j + 1)]), format="csr")


if __name__ == "__main__":
    n = 1001
    A = get_skew_symmetric_matrix(int((n-1)/2))
    b = np.ones(n) / np.linalg.norm(np.ones(n))


    exact, eigvals, _ = np_funm_krylov(A, b, {"restart_length": 1000, "num_restarts": 1})
    for restart_length in [16, 32, 64]:
        num_restarts = 200 // restart_length + 1
        param = {
            "restart_length": restart_length,
            "num_restarts": num_restarts
        }

        # Calculate the matrix exponential
        start = time.time()
        npfs, npeigvals, npupdate_norms = np_funm_krylov(A, b, param)
        print(time.time() - start)
        npnorms = list(np.linalg.norm(exact - npfs, axis=0))
        plt.plot(restart_length * np.arange(1, param["num_restarts"] + 1), npnorms, label=f"m:{restart_length}")
        #plt.scatter(np.arange(1, param["num_restarts"] + 1), npupdate_norms, label=f"m:{restart_length}")
    plt.title("Error of restarted expm for skew symmetric matrix")
    plt.legend(framealpha=.5)
    plt.yscale("log")
    plt.ylim(bottom=max(np.finfo(npfs.dtype).eps, plt.ylim()[0]) / 10)
    plt.xlabel("Arnoldi iterations")
    plt.savefig(os.path.join(root_path, f"figures/skewSym{n}ErrorPlot.png"))
    plt.show()

    plt.title("Ritz values of skew symmetric matrix")
    plt.scatter(np.real(eigvals[0]), np.imag(eigvals[0]), marker="o", label="unrestarted")
    for k in [param["num_restarts"] - 1]:
        plt.scatter(np.real(npeigvals[k]), np.imag(npeigvals[k]), marker="x", label=(k + 1) * restart_length, s=20 - k)

    plt.legend()
    plt.ylabel("Imaginary")
    plt.xlabel("Real")
    plt.savefig(os.path.join(root_path, f"figures/skewSym{n}RitzValues.png"))
    plt.show()
