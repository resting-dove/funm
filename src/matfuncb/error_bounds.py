import numpy as np
import scipy.sparse
from typing import Union


def gershgorin(A: np.array):
    """Approximate an interval that the Eigenvalues of the matrix A lie in
    by the Gershgorin circles.
    For now assume that the matrix is symmetric and thus the spectrum
    entirely real."""
    a, b = 0, 0
    n = A.shape[0]
    for i in range(n):
        center = A[i, i]
        radius = np.sum(np.abs(A[i, range(n) != i]))
        a = min(a, center - radius)
        b = max(b, center + radius)
    return a, b


def expm_error_bound(rho: float, stopping_acc: float):
    # TODO: Where does this really come from
    # TODO: Adjust to other intervals than just [0,b]
    m = int(np.sqrt(-np.log(stopping_acc / 10) * 5 * rho))
    return m


def get_length_gershgorin(A: Union[np.array, scipy.sparse.sparray], stopping_acc: float):
    low, high = gershgorin(A)
    print(f"Largest eigenvalue estimated to be {high}.")
    print("For now assume the spectrum of A is entirely real and positive.")
    rho = high / 4
    # Use the first error bound from the paper
    m = expm_error_bound(rho, stopping_acc)
    assert np.sqrt(4 * rho) <= m
    assert m <= 2 * rho
    return m


def power_method(A: np.array, b: np.array, iterations: int, tol=1e-14):
    """Perform a certain number of iterations of the power method."""
    converged = False
    i = 0
    prev = 0
    nextb = A.dot(b)
    while not converged and i < iterations:
        i += 1
        b = nextb
        b = b / np.linalg.norm(b)
        nextb = A.dot(b)
        eig = b.T @ nextb / np.linalg.norm(b) ** 2
        if np.abs(eig - prev) < tol:
            break
        prev = eig
    return eig


def get_length_power(A: Union[np.array, scipy.sparse.sparray], b: np.array, stopping_acc: float):
    high = power_method(A, b, 3)
    print(f"Largest eigenvalue estimated to be {high}.")
    print("For now assume the spectrum of A is entirely real and positive.")
    rho = high / 4
    # Use the first error bound from the paper
    m = int(np.sqrt(-np.log(stopping_acc / 10) * 5 * rho))
    assert np.sqrt(4 * rho) <= m
    assert m <= 2 * rho
    return m


def hochbruck_lubich(sm_ev: float, t: float, n: int):
    """
    A priori error bound for the Arnoldi approximation of exp(tA),
    where the spectrum of A is entirely in the left half plane.

    Source:  Theorem 2 of  M. Hochbruck and C. Lubich,
    “On Krylov Subspace Approximations to the Matrix Exponential Operator,”
    SIAM J. Numer. Anal., vol. 34, no. 5, pp. 1911–1925, Oct. 1997, doi: 10.1137/S0036142995280572.

    Note: 2-norm."""
    assert sm_ev < 0
    rho = sm_ev / -4

    ms = np.arange(1, n + 1)
    out = np.empty_like(ms) * np.nan
    out = np.where(ms >= 2 * rho * t, 10 / (rho * t) / np.exp(rho * t) * (np.exp(1) * rho * t / ms) ** ms, out)
    indices = (2 * rho * t >= ms) * (ms >= np.sqrt(4 * rho * t))
    out = np.where(indices, 10 / np.exp(ms ** 2 / (5 * rho * t)), out)
    return ms, out


def saad(norm_A: float, t: float, n: int):
    """
    A priori error bound for the Lanczos approximation of exp(tA),
    where t||A||<1.
    Output is in norm compatible with input norm.

    Source [1] Y. Saad, “Analysis of Some Krylov Subspace Approximations to the Matrix Exponential Operator,”
     SIAM J. Numer. Anal., vol. 29, no. 1, pp. 209–228, Feb. 1992, doi: 10.1137/0729014.
."""

    ms = np.arange(1, n)
    norm = t * norm_A
    out = [2 * norm * np.exp(norm)]
    for m in ms[1:]:
        out.append(out[-1] * norm / m)
    return ms, np.array(out)


def ye_entry_1(m: np.array, t: float, sm_eval: float, la_eval: float, q=0.4):
    assert sm_eval <= la_eval
    b = la_eval
    a = sm_eval
    q0 = (np.sqrt(b) - np.sqrt(a)) / (np.sqrt(b) + np.sqrt(a))
    gamma = (b - a) * (q - q0) * (1 / q0 - q)
    return 2 / (1 - q) * np.exp(-t * gamma / 4 / q) * q ** (m - 1)


def ye_entry_2(m: np.array, t: float, la_eval: float):
    b = la_eval
    return 1 / (scipy.special.factorial(m - 1)) * (t * b / 2) ** (m - 1)


def ye(sm_eval: float, la_eval: float, t: float, n: int, alpha: float):
    """
    A priori error bound for the Lanczos approximation of exp(tA),
    where A is symmetric positive semi definite.

    Arguments:
        sm_eval and la_eval are estimates for the smallest and largest eigenvalues of A.
        t is the parameter in exp(tA)
        n is the size of A
        alpha is a float in [0,t]

    Source: Theorem 1 in Q. Ye, “Error Bounds for the Lanczos Methods for Approximating Matrix Exponentials,”
    SIAM J. Numer. Anal., vol. 51, no. 1, pp. 68–87, Jan. 2013, doi: 10.1137/11085935X.
    """
    assert alpha <= t
    m = np.arange(1, n)
    beta = 1  # This could be anything really but doesn't change shape, it's actually supposed to come from the Lanczos decomp
    h0 = ye_entry_1(m, 0, sm_eval, la_eval)
    hhalf = ye_entry_1(m, alpha / 2, sm_eval, la_eval)
    halpha = ye_entry_1(m, alpha, sm_eval, la_eval)
    hthreequart = ye_entry_2(m, alpha + (t - alpha) / 2, la_eval)
    hthreequart2 = ye_entry_2(m, alpha + (t - alpha) / 2, la_eval)
    ht = ye_entry_1(m, t, sm_eval, la_eval)
    ht2 = ye_entry_2(m, t, la_eval)
    left = np.maximum(h0, hhalf, halpha)
    right = np.minimum(np.maximum(halpha, hthreequart, ht), np.maximum(hthreequart2, ht2))
    # result = beta * (h0alpha * np.exp((alpha - t) * lambda_min) * alpha + ye_entry_2(m, t, sm_eval) * (t - alpha))
    result = beta * (left * np.exp((alpha - t) * sm_eval) * alpha + right * (t - alpha))
    return m, result


def get_points_on_circle(c: float, r: float, n: int):
    theta = np.linspace(0, 2 * np.pi, n)
    return c + r * (np.cos(theta) + 1j * np.sin(theta))


def chen_musco(sm_eval: float, la_eval: float, w: float, n: int, f=np.exp):
    """
    A prioir error bound for Lanczos approximation of f(A).
    Basically shift to the system (A-wI) and use an error bound for the linear system.
    In this case use a bound for the k-th CG step of (A-wI)x=b.
    Also multiply by max of f over D(sm_eval, spread + w).

    Source: Corrolary 3.1 of [1] T. Chen, A. Greenbaum, C. Musco, and C. Musco,
    “Error Bounds for Lanczos-Based Matrix Function Approximation,”
    SIAM J. Matrix Anal. Appl., vol. 43, no. 2, pp. 787–811, Jun. 2022, doi: 10.1137/21M1427784.
    """
    assert sm_eval <= la_eval
    assert w != sm_eval  # needs to be outside the spectrum of T
    m = np.arange(1, n)
    is_pos_def = None

    spectrum_spread = np.abs(la_eval - sm_eval)
    kappa = np.abs(la_eval - w) / np.abs(sm_eval - w)
    if kappa < 1:
        is_pos_def = False
        kappa = 1 / kappa
        center = la_eval
        radius = np.abs(la_eval - w)
    else:
        is_pos_def = True
        center = sm_eval
        radius = np.abs(sm_eval - w)
    sqrt_kappa = np.sqrt(kappa)
    integral_part = radius * np.max(
        np.abs(f(get_points_on_circle(center, radius, 20))))
    cg_bound = 2 * ((sqrt_kappa - 1) / (sqrt_kappa + 1)) ** m
    # cg_bound = 2 * np.exp(- 2 * m / sqrt_kappa)
    result = integral_part * cg_bound
    return m, result


def h_w_z(x, w, z):
    return (x - w) / (x - z)


def bound_h_w_z(w, z, a, b):
    """Bound the expression (x - w) / (x - z) on the interval [a,b]."""
    x_star = (np.real(z) ** 2 + np.imag(z) ** 2 - np.real(z) * w)
    result = np.zeros_like(z)
    np.divide((z - w), np.imag(z), out=result,
              where=(a * (np.real(z) - w) <= x_star) * (x_star <= b * (np.real(z) - w)))
    # result = np.where((a * (np.real(z) - w) <= x_star) * (x_star <= b * (np.real(z) - w)),
    #                   np.abs((z - w) / np.imag(z)), 0)
    result = np.maximum(np.maximum(np.abs((a - w) / (a - z)), np.abs((b - w) / (b - z))), result)
    return result


def get_determinant_tridiag(T, shift=0.0):
    n = T.shape[0]
    assert n > 1
    # f3 = 0  # f_{n-3}
    f = [1]
    f.append(T[0, 0] - shift)
    for i in range(1, n):
        a, b, c = T[i, i] - shift, T[i - 1, i], T[i, i - 1]
        fi = a * f[-1] - b * c * f[-2]
        f.append(fi)
    return f[-1]


def chen_musco_post(T: np.array, w: float, f=np.exp, fix_0_eval=True):
    """
    A posteriori error bound for Lanczos approximation of f(A).
    In comparison to the a priori bound the tridiagonal matrix T is available.

    Source: Section 3.2 T. Chen, A. Greenbaum, C. Musco, and C. Musco,
    “Error Bounds for Lanczos-Based Matrix Function Approximation,”
    SIAM J. Matrix Anal. Appl., vol. 43, no. 2, pp. 787–811, Jun. 2022, doi: 10.1137/21M1427784.
    """
    ritz = np.sort(scipy.linalg.eigvals(T))
    sm_eval = ritz[0]
    la_eval = ritz[-1]
    assert w < sm_eval or w > la_eval
    kappa = (la_eval - w) / (sm_eval - w)
    if 0 < kappa < 1:
        kappa = 1 / kappa
        if fix_0_eval:
            la_eval = 0
        c, r = sm_eval, np.abs(sm_eval - w)
    elif kappa < 0:
        raise RuntimeError("this shouldn't happen")
    else:
        if fix_0_eval:
            sm_eval = 0
        c, r = la_eval, np.abs(la_eval - w)


    a2c = lambda theta: c + r * (np.cos(theta) + 1j * np.sin(theta))
    dr = lambda theta: np.abs(r * (-np.sin(theta) + 1j * np.cos(theta)))
    F = lambda theta: np.abs(f(a2c(theta)))
    H_w_z = lambda theta: bound_h_w_z(w, a2c(theta), sm_eval, la_eval)
    Dets = lambda theta: np.abs(get_determinant_tridiag(T, w) / get_determinant_tridiag(T, a2c(theta)))
    integrand = lambda theta: F(theta) * H_w_z(theta) * Dets(theta) * dr(theta)
    integral, abserr = scipy.integrate.quad(integrand, 0, 2 * np.pi, epsabs=10)
    if abserr > integral:
        print(f"integral: {integral}, abserr: {abserr}")

    sqrt_kappa = np.sqrt(kappa)
    m = T.shape[0]
    cg_bound = 2 * ((sqrt_kappa - 1) / (sqrt_kappa + 1)) ** m
    #cg_bound = 2 * np.exp(- 2 * m / sqrt_kappa)
    return m, integral * cg_bound

def afanasjew_post(T: np.array, V:np.array, A:np.array, m: int, f: callable):
    assert m < T.shape[0]
    delta = T[m+1, m]
    Tm = T[:m, :m]
    assert m < V.shape[1]
    v = V[:, m]
    ritz = np.real(np.sort(scipy.linalg.eigvals(Tm)))
    sm_eval = ritz[0]
    la_eval = ritz[-1]
    B = np.array([[sm_eval, 0], [1, la_eval]])
    E = np.zeros((2, Tm.shape[1]))
    E[0, -1] = 1
    H_tilde = np.block([[Tm, np.zeros((Tm.shape[0], 2))], [E, B]])
    fH = f(H_tilde)
    phi_1 = np.abs(fH[-2, 0])
    phi_2 = np.abs(fH[-1, 0])
    lower = phi_1 * v
    upper = lower + phi_2 * (A - la_eval * np.eye(A.shape[0])) @ v
    return np.linalg.norm(lower), np.linalg.norm(upper)



