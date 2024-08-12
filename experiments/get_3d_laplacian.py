import numpy as np
import scipy


def get_3d_laplacian(nx, ny, nz):
    """Construct the 3d Laplacian for homogenous heat equation.
    Source: https://www.mathworks.com/matlabcentral/fileexchange/27279-laplacian-in-1d-2d-or-3d"""
    ex = np.ones(nx)
    ey = np.ones(ny)
    ez = np.ones(nz)
    D1x = scipy.sparse.spdiags(np.array([-ex, 2 * ex, -ex]), [-1, 0, 1], nx, nx)
    D1y = scipy.sparse.spdiags(np.array([-ey, 2 * ey, -ey]), [-1, 0, 1], ny, ny)
    D1z = scipy.sparse.spdiags(np.array([-ez, 2 * ez, -ez]), [-1, 0, 1], nz, nz)
    Ix = scipy.sparse.eye(nx)
    Iy = scipy.sparse.eye(ny)
    Iz = scipy.sparse.eye(nz)
    A = (scipy.sparse.kron(Iz, scipy.sparse.kron(Iy, D1x)) +
         scipy.sparse.kron(Iz, scipy.sparse.kron(D1y, Ix)) +
         scipy.sparse.kron(scipy.sparse.kron(D1z, Iy), Ix))
    return A


def get_smallest_evs(m, nx, ny, nz):
    """ Calculate the m smallest exact eigenvalues of the 3d heat eq. with homogenous Dirichlet boundary conditions.
    Source: https://www.mathworks.com/matlabcentral/fileexchange/27279-laplacian-in-1d-2d-or-3d"""
    ex = np.ones(nx).reshape((-1, 1))
    ey = np.ones(ny).reshape((-1, 1))
    ez = np.ones(nz).reshape((-1, 1))
    if m > 0:
        a1 = np.pi / 2 / (nx + 1)
        N = np.arange(1, nx + 1).reshape((-1, 1))
        lambda1 = 4 * np.sin(a1 * N) ** 2

        a2 = np.pi / 2 / (ny + 1)
        N = np.arange(1, ny + 1).reshape((-1, 1))
        lambda2 = 4 * np.sin(a2 * N) ** 2

        a3 = np.pi / 2 / (nz + 1)
        N = np.arange(1, nz + 1).reshape((-1, 1))
        lambda3 = 4 * np.sin(a3 * N) ** 2

        lambd = (scipy.linalg.kron(ez, scipy.linalg.kron(ey, lambda1)) +
                 scipy.linalg.kron(ez, scipy.linalg.kron(lambda2, ex)) +
                 scipy.linalg.kron(lambda3, scipy.linalg.kron(ey, ex)))
        p = np.argsort(lambd)  # Positions of the largest EVs
        lambd = lambd[p.flatten()]
        if m < nx + ny + nz:
            w = lambd[m]
        else:
            w = np.inf

        lambd = lambd[:m]
        p = p[:m].T
        #######################
        p1 = np.mod(p, nx) + 1  # np.mod(p - 1, nx) + 1  # Rows (x-axis) of the largest EVs (1-indexed)

        V1 = np.sin(
            scipy.linalg.kron(np.arange(1, nx + 1).reshape((-1, 1)) * (np.pi / (nx + 1)), p1)
        ) * (2 / (nx + 1)) ** 0.5

        p2 = np.mod(p - p1 + 1, nx * ny)  # np.mod(p - p1, nx * ny)  # Which z-slice is the position in? (0-indexed)
        p3 = (p - p2 - p1 + 1) // (nx * ny) + 1  # z-slice of the largest EVs (1-indexed)
        p2 = p2 // nx + 1  # Column of the largest Evs (1-indexed)
        V2 = np.sin(scipy.sparse.kron(np.arange(1, ny + 1).reshape((-1, 1)) * (np.pi / (ny + 1)), p2)) * (
                2 / (ny + 1)) ** 0.5
        V3 = np.sin(scipy.sparse.kron(np.arange(1, nz + 1).reshape((-1, 1)) * (np.pi / (nz + 1)), p3)) * (
                2 / (nz + 1)) ** 0.5

        V = (scipy.sparse.kron(ez, scipy.sparse.kron(ey, V1)).multiply(
            scipy.sparse.kron(ez, scipy.sparse.kron(V2, ex)).multiply(
                scipy.sparse.kron(scipy.sparse.kron(V3, ey), ex))))

        if m != 0 and np.abs(lambd[-1] - w) < nx * ny * nz * np.finfo(float).eps:
            print('Warning: (m+1)th eigenvalue is  nearly equal to mth.')

        ########################

    else:
        lambd = []
        V = []

    return lambd, V
