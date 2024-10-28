import matplotlib.pyplot as plt
import scipy
import numpy as np
from utils import get_fig_ax, postprocess_style
import os

root_path = os.getcwd()

if __name__ == "__main__":
    n = 25  # Interior grid points in each direction
    N = n ** 3  # Total number of interior points
    h = 1 / (n + 1)
    # A = -1 / h ** 2 * get_3d_laplacian(n, n, n)
    lap = scipy.sparse.linalg.LaplacianNd((n, n, n), boundary_conditions='dirichlet')
    A = 1 / h ** 2 * lap.tosparse()
    t = 1
    evals = t / h ** 2 * lap.eigenvalues(N)

    nb = 50
    bins = np.linspace(-12 / h ** 2, 0, nb)
    d = bins[1] - bins[0]
    bins = [bins[0] - d] + list(bins) + [bins[-1] + d]
    print((evals <= bins[1]).sum())
    fig, ax = get_fig_ax(factor=0.6, ratio=21/9)

    ax.hist(evals, bins=bins)
    ax.set_ylabel('Count Eigenvalues')
    postprocess_style()
    fig.tight_layout()
    fig.savefig(os.path.join(root_path, f"figures/laplacian_{n}_spectrum.png"))
    fig.show()