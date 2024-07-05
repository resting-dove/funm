import numpy as np
import scipy

from Arnoldi import arnoldi


def evalnodal(x: np.array, nodes: np.array, subdiag: np.array):
    """
    Returns Prod(subdiag) / (x - nodes)
    :param x:
    :param nodes:
    :param subdiag:
    :return:
    """
    if len(nodes) != len(subdiag):
        raise RuntimeError("Evalnodal: nodes and subdiag need to be of same length.")
    p = np.ones_like(x)
    for j in range(len(nodes)):
        p = p * subdiag[j] / (x - nodes[j])
    return p


def is_upper_tri(A, tol=1e-6):
    return np.all(A[np.tril_indices(A.shape[0], -1)] < tol)


def eigs_2x2(A):
    """Source:
    https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_EigenProblem2.html#QR-Algorithm-for-computing-eigenvalues
    """
    b = -(A[-1, -1] + A[-2, -2])
    c = A[-1, -1] * A[-2, -2] - A[-2, -1] * A[-1, -2]
    d = np.sqrt(b ** 2 - 4 * c)

    if b > 0:
        return (-2 * c / (b + d), -(b + d) / 2)
    else:
        return ((d - b) / 2, 2 * c / (d - b))


def get_eigvals_qr(H, max_iterations=1000, tol=1e-6):
    """Calculate the eigenvalues of the matrix H using the QR algorithm.
    Influenced by
    https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_EigenProblem2.html#QR-Algorithm-for-computing-eigenvalues."""
    n = H.shape[0]
    eigvals = np.zeros(n)
    q, r = scipy.linalg.qr(H)
    A = np.dot(r, q)
    I = np.eye(n)
    for _ in range(max_iterations):
        if np.abs(A[-1, -2]) < tol:
            n -= 1
            eigvals[n] = A[-1, -1]
            A = A[:-1, :-1]
            I = np.eye(n)
        if n == 2:
            eigvals[:2] = eigs_2x2(A)
            break
        shift = A[-1, -1]
        q, r = scipy.linalg.qr(A - shift * I)
        A = np.dot(r, q) + shift * I
    #print(f"It took {i} iterations to get max entry to {np.max(np.tril(A))}.")
    return eigvals



# def evaluate_expm_quad(N: int, H: np.array, ritz_values: np.array, tol: float):
#     """
#     Evaluate expm(A)@b as a Midpoint rule quadrature.
#     :return:
#     """
#     m = H.shape[1]
#     aa = max(1, np.max(np.real(ritz_values)) + 1)
#     bb = 1
#     thetas = np.imag(ritz_values)
#     ccs = np.sqrt((aa + 1j * thetas - ritz_values))
#     cc = min(np.min(ccs) / 5, 0.25)
#
#     def gamma(theta):
#         return aa + 1j * bb * theta - cc * theta ** 2
#
#     def gamma_deriv(theta):
#         return 1j - 2 * cc * theta
#
#     thetac = np.sqrt((1 - np.log(tol)) / cc)
#
#     theta = np.linspace(-thetac, thetac, N)  # nodes
#     z = gamma(theta)
#     ee = np.eye(m, 1)
#     h = np.zeros((m, 1))
#     for j in range(N):
#         zeta_j = thetac * ((2 * j - 1) / N - 1)
#         numerator = np.exp(gamma(zeta_j)) * gamma_deriv(zeta_j)
#         mat = z[j] * scipy.sparse.eye(*H.shape) + H
#         r = scipy.linalg.solve(mat, ee)
#         h = h + numerator * r
#     h *= 2 * thetac / N
#     return h

def phi(theta: np.array, aa: float, bb: float, cc: float):
    return aa + 1j * bb * theta - cc * theta ** 2


def orig_quad(N: int, H: np.array, ritz_values: np.array, subdiag: np.array, tol: float):
    m = H.shape[1]

    aa = max(1, np.max(np.real(ritz_values)) + 1)
    bb = 1
    thetas = np.imag(ritz_values)
    ccs = np.abs((ritz_values - aa - 1j * thetas) / thetas ** 2)
    cc = min(np.min(ccs) / 5, 0.25)

    thetac = np.sqrt((aa - np.log(tol)) / cc)
    theta = np.linspace(-thetac, thetac, N)
    hh = theta[1] - theta[0]
    z = phi(theta, aa, bb, cc)
    c = -hh / (2j * np.pi) * np.exp(z) * (1j * bb - 2 * cc * theta)

    tt = z[: N // 2]
    rho_vec = evalnodal(tt, ritz_values, subdiag)

    h1 = np.zeros((m, 1))
    c = c[: N // 2] * rho_vec
    for j in range(N // 2):
        #mat = z[j] * scipy.sparse.eye(*H.shape) - H
        mat = z[j] * np.eye(*H.shape) - H
        # TODO: Does this realize these are systems of shifted LS?
        r = scipy.linalg.solve(mat, np.eye(m, 1))
        h1 = h1 - c[j] * r
    h1 = 2 * np.real(h1)
    return h1


def expm_quad(A: scipy.sparse.sparray, b: np.array, max_restarts: int, restart_length: int, arnoldi_trunc=np.inf,
              tol=1e-10, stopping_acc=1e-10, keep_H=False, keep_V=False, keep_f=False, max_refinements=20):
    """
    Calculate expm(A)@b using a restarted Krylov method based on quadrature.
    The code is inspired by Matlab code from A. Frommer, S.G\"{u}ttel and M. Schweitzer.
    :param A: scipy.sparse.sparray of size nxn.
    :param b: np.array of size nx1.
    :param max_restarts: int.
    :param restart_length: int.
    :param arnoldi_trunc: int, Number of vectors to orthogonalize against in Arnoldi subroutine.
    :param tol: float, quadrature is deemed accurate, when increasing number of nodes changes output less than this.
    :param stopping_acc: float, restarts stop when norm of update is below this number.
    :return: f: np.array of size nx1 the result of expm(A)@b.
    """
    stop_condition = False
    beta = np.sqrt(np.inner(b, b))
    v = b / beta
    n = len(b)

    if restart_length > n:
        print("Restart length too long")
        restart_length = n
        max_restarts = 1

    m = restart_length
    subdiag = np.array([])
    f = np.zeros_like(b)
    if keep_H:
        full_H = np.zeros((m * max_restarts + 1, m * max_restarts + 1))
    if keep_V:
        full_V = np.zeros((n, m * max_restarts + 2))
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
        # TODO: Does this use the fact it's a Hessenberg?
        #interpol_nodes[k] = scipy.linalg.eig(H, right=False)
        interpol_nodes[k] = get_eigvals_qr(H)

        active_nodes = np.array([])
        for kk in range(k):
            active_nodes = np.concatenate((active_nodes, interpol_nodes[kk]))

        if k == 0:
            h2 = scipy.linalg.expm(H)
        else:
            converged = False
            h1 = np.array([])
            for _ in range(max_refinements):
                if converged:
                    break
                if len(h1) == 0:
                    if N > 2:
                        N = int(N / np.sqrt(2))
                    N -= N % 2
                    h1 = orig_quad(N, H, active_nodes, subdiag, tol)
                # Increase the number of quadrature points
                # If quadrature did not converge, loop starts again here.
                N2 = int(np.sqrt(2) * N) + 1
                N2 += N2 % 2
                h2 = orig_quad(N2, H, active_nodes, subdiag, tol)
                # Check for convergence
                if np.linalg.norm(beta * (h2 - h1)) / np.linalg.norm(f) < tol:
                    converged = True  # print(f"{N} quadrature points were enough.")
                else:
                    # print(f"{N} quadrature points were not enough. Trying {N2}.")
                    h1 = h2
                    N = N2
        h_big = beta * h2[:m, 0]
        f = V @ h_big + f
        update = np.linalg.norm(h_big)

        if m != 1:
            if len(subdiag) > 0:
                subdiag = np.concatenate((subdiag, np.diag(H, -1), [eta]))
            else:
                subdiag = np.concatenate((np.diag(H, -1), [eta]))
        else:
            if len(subdiag) > 0:
                subdiag = np.concatenate((subdiag, [eta]))
            else:
                subdiag = np.array([eta])
        if update / np.linalg.norm(f) < stopping_acc:
            stop_condition = True
            print("Stopping accuracy reached.")
        if k > 10 and (update / out["update"][-1] < .05):
            stop_condition = True
            print("Updates getting to small.")
        if keep_f:
            out["f"].append(f)
        out["update"].append(update)
        out["N"].append(N)

    return f, out
