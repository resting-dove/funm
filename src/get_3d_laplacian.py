import scipy
import numpy as np


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
    A = (scipy.sparse.kron(Iz,
                           scipy.sparse.kron(Iy, D1x)) +
         scipy.sparse.kron(Iz, scipy.sparse.kron(D1y, Ix)) +
         scipy.sparse.kron(scipy.sparse.kron(D1z, Iy), Ix)
         )
    return A
