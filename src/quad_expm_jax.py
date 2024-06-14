# In this file I want to make a minimal poc of a jax implementation of a quadrature based restarted Krylov method for
# exp(A)b.
# 1. Just write numpy -> jax without jitting or lax
# 2. Speed up embarrassingly parallel parts
# 3. Jit some subroutines

from functools import partial

import jax
import numpy as np
import scipy
from jax import numpy as jnp
from jax_Arnoldi import arnoldi_jittable

import numpy as np
import scipy

from Arnoldi import arnoldi


def evalnodal(x: jax.Array, nodes: jax.Array, subdiag: jax.Array):
    """
    Returns Prod(subdiag) / (x - nodes)
    :param x:
    :param nodes:
    :param subdiag:
    :return:
    """
    if len(nodes) != len(subdiag):
        raise RuntimeError("Evalnodal: nodes and subdiag need to be of same length.")
    p = jnp.ones_like(x)
    for j in range(len(nodes)):
        p = p * subdiag[j] / (x - nodes[j])
    return p


def is_upper_tri(A: jax.Array, tol=1e-6):
    return jnp.all(A[jnp.tril_indices(A.shape[0], -1)] < tol)


def eigs_2x2(A: jax.Array):
    """Source:
    https://www.johndcook.com/blog/2021/05/07/trick-for-2x2-eigenvalues/
    """
    m = (A[-1, -1] + A[-2, -2]) / 2
    p = A[-1, -1] * A[-2, -2] - A[-2, -1] * A[-1, -2]
    d = jnp.sqrt(m ** 2 - p)
    return jnp.array([m + d, m - d])


def get_eigvals_qr(H: jax.Array, max_iterations=1000, tol=1e-6):
    """Calculate the eigenvalues of the matrix H using the QR algorithm.
    Influenced by
    https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_EigenProblem2.html#QR-Algorithm-for-computing-eigenvalues."""
    n = H.shape[0]
    eigvals = np.zeros(n)
    q, r = jax.scipy.linalg.qr(H)
    A = jnp.dot(r, q)
    I = jnp.eye(n)
    for _ in range(max_iterations):
        if jnp.abs(A[-1, -2]) < tol:
            n -= 1
            eigvals[n] = A[-1, -1]
            A = A[:-1, :-1]
            I = jnp.eye(n)
        if n == 2:
            eigvals[:2] = eigs_2x2(A)
            break
        shift = A[-1, -1]
        q, r = jax.scipy.linalg.qr(A - shift * I)
        A = jnp.dot(r, q) + shift * I
    #print(f"It took {i} iterations to get max entry to {np.max(np.tril(A))}.")
    return eigvals




def phi(theta: jax.Array, aa: float, bb: float, cc: float):
    return aa + 1j * bb * theta - cc * theta ** 2


def orig_quad(N: int, H: jax.Array, ritz_values: jax.Array, subdiag: jax.Array, tol: float):
    m = H.shape[1]

    aa = max(1, jnp.max(jnp.real(ritz_values)) + 1)
    bb = 1
    thetas = jnp.imag(ritz_values)
    ccs = jnp.abs((ritz_values - aa - 1j * thetas) / thetas ** 2)
    cc = jax.lax.clamp(0.25, jnp.nanmin(jnp.min(ccs) / 5, initial=0.25), jnp.inf)

    thetac = jnp.sqrt((aa - jnp.log(tol)) / cc)
    theta = jnp.linspace(-thetac, thetac, N)
    hh = theta[1] - theta[0]
    z = phi(theta, aa, bb, cc)
    c = -hh / (2j * jnp.pi) * jnp.exp(z) * (1j * bb - 2 * cc * theta)

    tt = z[: N // 2]
    rho_vec = evalnodal(tt, ritz_values, subdiag)

    h1 = jnp.zeros((m, 1))
    c = c[: N // 2] * rho_vec
    for j in range(N // 2):
        mat = z[j] * jnp.eye(*H.shape) - H
        # TODO: Does this realize these are systems of shifted LS?
        r = jax.scipy.linalg.solve(mat, jnp.eye(m, 1))
        h1 = h1 - c[j] * r
    h1 = 2 * jnp.real(h1)
    return h1


def expm_quad(A: jax.Array, b: jax.Array, max_restarts: int, restart_length: int, arnoldi_trunc=jnp.inf,
              tol=1e-8, stopping_acc=1e-8, keep_H=False, keep_V=False, keep_f=False):
    """
    Calculate expm(A)@b using a restarted Krylov method based on quadrature.
    The code is inspired by Matlab code from A. Frommer, S.G\"{u}ttel and M. Schweitzer.
    :param A: jax.Array of size nxn.
    :param b: jax.Array of size nx1.
    :param max_restarts: int.
    :param restart_length: int.
    :param arnoldi_trunc: int, Number of vectors to orthogonalize against in Arnoldi subroutine.
    :param tol: float, quadrature is deemed accurate, when increasing number of nodes changes output less than this.
    :param stopping_acc: float, restarts stop when norm of update is below this number.
    :return: f: jax.Array of size nx1 the result of expm(A)@b.
    """
    stop_condition = False
    beta = jnp.sqrt(jnp.inner(b, b))
    v = b / beta
    n = len(b)

    if restart_length > n:
        print("Restart length too long")
        restart_length = n
        max_restarts = 1

    m = restart_length
    subdiag = jnp.array([])
    f = jnp.zeros_like(b)
    if keep_H:
        full_H = jnp.zeros((m * max_restarts + 1, m * max_restarts + 1))
    if keep_V:
        full_V = jnp.zeros((n, m * max_restarts + 2))
    out = {}
    out["f"] = []
    out["update"] = []
    out["N"] = []
    N = 32  # Number of quadrature nodes
    interpol_nodes = {}
    for k in range(max_restarts):
        if stop_condition:
            break
        (v, V, H, breakdown) = arnoldi(A=A, w=v, m=m, trunc=arnoldi_trunc)
        if breakdown:
            stop_condition = True
            print("Arnoldi breakdown occured.")
        if keep_H:
            full_H[k * m: (k + 1) * m + 1, k * m: (k + 1) * m] = H
        if keep_V:
            full_V[:, k * m: (k + 1) * m] = V
        eta = H[-1, -1]
        H = H[:m, :m]
        #interpol_nodes[k] = scipy.linalg.eig(H, right=False)
        interpol_nodes[k] = get_eigvals_qr(H)

        active_nodes = jnp.array([])
        for kk in range(k):
            active_nodes = jnp.concatenate((active_nodes, interpol_nodes[kk]))

        if k == 0:
            h2 = jax.scipy.linalg.expm(H)
        else:
            converged = False
            h1 = jnp.array([])
            while not converged:
                if len(h1) == 0:
                    if N > 2:
                        N = int(N / jnp.sqrt(2))
                    N -= N % 2
                    h1 = orig_quad(N, H, active_nodes, subdiag, tol)
                # Increase the number of quadrature points
                # If quadrature did not converge, loop starts again here.
                N2 = (jnp.sqrt(2) * N).astype(int) + 1
                N2 += N2 % 2
                h2 = orig_quad(N2, H, active_nodes, subdiag, tol)
                # Check for convergence
                if jnp.linalg.norm(beta * (h2 - h1)) / jnp.linalg.norm(f) < tol:
                    converged = True  # print(f"{N} quadrature points were enough.")
                else:
                    # print(f"{N} quadrature points were not enough. Trying {N2}.")
                    h1 = h2
                    N = N2
        h_big = beta * h2[:m, 0]
        f = V @ h_big + f
        update = beta * jnp.linalg.norm(h_big)

        if m != 1:
            if len(subdiag) > 0:
                subdiag = jnp.concatenate((subdiag, jnp.diag(H, -1), jnp.array([eta])))
            else:
                subdiag = jnp.concatenate((jnp.diag(H, -1), jnp.array([eta])))
        else:
            if len(subdiag) > 0:
                subdiag = jnp.concatenate((subdiag, jnp.array([eta])))
            else:
                subdiag = jnp.array([eta])
        if update / jnp.linalg.norm(f) < stopping_acc:
            stop_condition = True
            print("Updates getting too small.")
        if k > 10 and (update / out["update"][-1] < .05):
            stop_condition = True
            print("Updates stagnate.")
        if keep_f:
            out["f"].append(f)
        out["update"].append(update)
        out["N"].append(N)

    return f, out
