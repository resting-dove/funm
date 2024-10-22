import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
import subprocess

from src.matfuncb.np_funm import lanczos_method
from src.matfuncb.krylov_basis import arnoldi
from src.matfuncb.error_bounds import *

root_path = os.getcwd()


def get_index(final_size: int, krylov_size: int) -> list:
    idx = [krylov_size * i for i in range((final_size + 1) // krylov_size)]
    if idx[-1] != final_size:
        idx += [final_size]
    return idx


def prepare_starting_vector(evecs, n: int):
    index = np.linspace(1, n, n)
    i, j, k, iprime, jprime, kprime = np.meshgrid(index, index, index, index, index, index, indexing='ij')
    factor = iprime + jprime + kprime
    evecs = evecs.reshape((evecs.shape[0], n, n, n))
    ret = (evecs / factor.reshape((n ** 3, n, n, n))).sum(axis=-1).sum(axis=-1).sum(axis=-1)
    return ret.reshape((n, n, n))


def prepare_starting_vector2(evecs, n: int):
    evecs = scipy.sparse.eye(evecs.shape[0], 1)  # evecs.sum(axis=-1)
    #return evecs / scipy.linalg.norm(evecs)
    return (evecs / scipy.sparse.linalg.norm(evecs)).todense()


def get_lanczos_errors(V, H, beta, exact, matfunc, step=1, upper=100):
    m = H.shape[1]
    idx = np.arange(0, min(m + 1, upper), step)
    errors = np.zeros(len(idx))
    errors[0] = np.linalg.norm(exact)

    for idxx, i in enumerate(idx[1:], start=1):
        H_exp = matfunc(H[:i, :i])
        H_exp_jax = H_exp[:, [0]]
        f = beta * (V[:, :i] @ H_exp_jax)
        errors[idxx] = np.linalg.norm(f.flatten() - exact)
    return idx, errors


if __name__ == "__main__":
    plot_store = {}
    plot_store["filename"] = os.path.basename(__file__)
    plot_store["git_commit"] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

    n = 5  # Interior grid points in each direction
    N = n ** 3  # Total number of interior points
    h = 1 / (n + 1)
    # A = -1 / h ** 2 * get_3d_laplacian(n, n, n)
    lap = scipy.sparse.linalg.LaplacianNd((n, n, n), boundary_conditions='dirichlet')
    A = 1 / h ** 2 * lap.tosparse()
    t = 0.1
    evals = t / h ** 2 * lap.eigenvalues(N)
    print("evals gotten")
    # if f"evecs_heat_{n}_{t}.npy" in os.listdir(os.path.join(root_path, "precalculated")):
    #     evecs = np.load(os.path.join(os.path.join(root_path, "precalculated"), f"evecs_heat_{n}_{t}.npy"))
    # else:
    #     evecs = lap.eigenvectors(N)
    #     np.save(os.path.join(os.path.join(root_path, "precalculated"), f"evecs_heat_{n}_{t}.npy"), evecs)
    # print("evecs gotten")
    evecs = scipy.sparse.eye(N**2)

    plot_store['n'] = n
    plot_store['t'] = t

    u0 = prepare_starting_vector2(evecs, n)
    print("prepared u0")
    exact = evecs.T @ u0.flatten()
    print("first matvec")
    exact = np.diagflat(np.exp(evals)) @ exact
    exact = (evecs @ exact)
    print("calculated exact")
    evecs = None  # Maybe this frees space
    print("killed evecs")
    fig, ax = plt.subplots()
    colors = plt.get_cmap('Set1').colors
    i = 0
    # for i, krylov_size in enumerate([20, 10, 6]):
    #     num_starts = 550 // krylov_size + 1
    #
    #     # Calculate the matrix exponential
    #     npfs, npupdate_norms, final_size = lanczos_method(t * A, u0.flatten(), scipy.sparse.linalg.expm,
    #                                                       krylov_size=krylov_size, max_starts=num_starts,
    #                                                       stopping_acc=-np.inf, arnoldi_acc=-np.inf,
    #                                                       stopping_decay=-np.inf)
    #     error_norms = [np.linalg.norm(exact - 0)] + list(np.linalg.norm(exact.reshape((-1, 1)) - npfs, axis=0))
    #     idx = get_index(final_size, krylov_size)
    #     name = f"m:{krylov_size}"
    #     plot_store[name + " error_norms"] = error_norms
    #     plot_store[name + " idx"] = idx
    #     plot_store[name + " update_norms"] = npupdate_norms
    #     line, = ax.plot(idx, error_norms, label=name, marker=".", linestyle="solid", c=colors[i])
    #     line, = ax.plot(idx[1:], npupdate_norms, linestyle="--", c=colors[i])

    i += 1
    ms, bounds = hochbruck_lubich(min(evals), 1, n=550)
    plot_store["Hochbruck Lubich ms"] = ms
    plot_store["Hochbruck Lubich bounds"] = bounds
    ax.plot(ms, bounds, label="HL", linestyle="--", c=colors[i])

    i += 1
    ms, bounds = saad(np.abs(min(evals)), 1, n=550)
    plot_store["Saad ms"] = ms
    plot_store["Saad bounds"] = bounds
    ax.plot(ms, bounds, label="S", linestyle="--", c=colors[i])

    i += 1
    ms, bounds = chen_musco(min(evals), max(evals), w=1, n=550, f=np.exp)
    plot_store["Chen Musco ms"] = ms
    plot_store["Chen Musco bounds"] = bounds
    ax.plot(ms, bounds, label="CGMM", linestyle="--", c=colors[i])

    i += 1
    ms, bounds = ye(min(evals), max(evals), t=1, n=550, alpha=0)
    plot_store["Ye alpha 0 ms"] = ms
    plot_store["Ye alpha 0 bounds"] = bounds
    ax.plot(ms, bounds, label="Ye 0", linestyle="-", c=colors[i])

    ms, bounds = ye(min(evals), max(evals), t=1, n=550, alpha=0.5)
    plot_store["Ye alpha 0.5 ms"] = ms
    plot_store["Ye alpha 0.5 bounds"] = bounds
    ax.plot(ms, bounds, label="Ye 0.5", linestyle="--", c=colors[i])

    ms, bounds = ye(min(evals), max(evals), t=1, n=550, alpha=1)
    plot_store["Ye alpha 1 ms"] = ms
    plot_store["Ye alpha 1 bounds"] = bounds
    ax.plot(ms, bounds, label="Ye 1", linestyle="-.", c=colors[i])

    beta = np.linalg.norm(u0.flatten())
    (w, V, H, m) = arnoldi(t * A, u0.flatten() / beta, N, trunc=1)
    print("Arnoldi finished")
    idx, lanczos_errors = get_lanczos_errors(V, scipy.sparse.csc_array(H), beta, exact, scipy.sparse.linalg.expm, 20,
                                             upper=300)
    plot_store["Lanczos idx"] = idx
    plot_store["Lanczos error_norms"] = lanczos_errors
    lanc_plot = ax.plot(idx, lanczos_errors, color='black', label=r"m=$\infty$")

    np.savez(os.path.join(root_path, "artifacts", "plot_store" + f"heat{n}_bounds"), **plot_store)

    ax.set_title(f"Heat equation with n={n}")
    ax.set_yscale("log")
    ax.set_ylim(bottom=max(np.finfo(evals[0].dtype).eps, plt.ylim()[0]) / 1000, top=10)
    ax.set_xlabel("Lanczos iterations")
    ax.set_ylabel("Error")
    ax.legend(framealpha=.5, scatterpoints=1, numpoints=1)
    fig.savefig(os.path.join(root_path, f"figures/heat{n}_bounds_ErrorPlot.png"))
    plt.show()
    1 + 1
