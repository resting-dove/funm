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
    A priori error bound for the Lanczos approximation of exp(tA).
    Output is in norm compatible with input norm.

    Source Theorem 4.3 Y. Saad, “Analysis of Some Krylov Subspace Approximations to the Matrix Exponential Operator,”
     SIAM J. Numer. Anal., vol. 29, no. 1, pp. 209–228, Feb. 1992, doi: 10.1137/0729014.
.   """
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


def chen_musco(sm_eval: float, la_eval: float, w: float, n: int, f=np.exp, *, center=None, radius=None):
    """
    A priori error bound for Lanczos approximation of f(A).
    Basically shift to the system (A-wI) and use an error bound for the linear system.
    In this case use a bound for the k-th CG step of (A-wI)x=b.
    Also multiply by max of f over D(sm_eval, spread + w).

    Source: Corrolary 3.3 of T. Chen, A. Greenbaum, C. Musco, and C. Musco,
    “Error Bounds for Lanczos-Based Matrix Function Approximation,”
    SIAM J. Matrix Anal. Appl., vol. 43, no. 2, pp. 787–811, Jun. 2022, doi: 10.1137/21M1427784.
    """
    m = np.arange(1, n)
    center, radius, kappa = setup_circle_and_kappa(sm_eval, la_eval, w, center, radius)
    integral_part = np.abs(radius) * np.max(
        np.abs(f(get_points_on_circle(center, radius, 20))))
    cg_bound = get_cg_bound(kappa, m)
    result = integral_part * cg_bound
    return m, result


def setup_circle_and_kappa(sm_eval: float, la_eval: float, w: float, center=None, radius=None, fix_0_eval=False):
    assert sm_eval <= la_eval
    assert w < sm_eval or w > la_eval  # needs to be outside the spectrum of T
    kappa = np.abs(la_eval - w) / np.abs(sm_eval - w)
    if 0 < kappa < 1:
        if fix_0_eval: kappa = np.abs(-w) / np.abs(sm_eval - w)
        kappa = 1 / kappa
        if not center and not radius:
            center = sm_eval
            radius = np.abs(sm_eval - w)
    elif kappa < 0:
        raise RuntimeError(f"After shifting by {w} {sm_eval} and {la_eval} should have the same sign.")
    else:
        if fix_0_eval: kappa = np.abs(la_eval - w) / np.abs(-w)
        if not center and not radius:
            center = la_eval
            radius = np.abs(la_eval - w)
    return center, radius, kappa


def chen_musco_no_kappa(A, b, sm_eval: float, la_eval: float, w: float, n: int, S: list, f=np.exp, *, center=None,
                        radius=None):
    """
    A theoretical error bound for the Lanczos approximation of f(A).
    This makes use of actual step m errors of the CG method.

    Source: Theorem 2.6 of T. Chen, A. Greenbaum, C. Musco, and C. Musco,
    “Error Bounds for Lanczos-Based Matrix Function Approximation,”
    SIAM J. Matrix Anal. Appl., vol. 43, no. 2, pp. 787–811, Jun. 2022, doi: 10.1137/21M1427784.
    """
    assert len(S) == 2  # For now just allow one interval [a,b]
    m = np.arange(1, n)
    center, radius, kappa = setup_circle_and_kappa(sm_eval, la_eval, w, center, radius)

    cg_bound = get_cg_errors(A, w, b, len(m))

    a2c = lambda theta: center + radius * (np.cos(theta) + 1j * np.sin(theta))
    dr = lambda theta: np.abs(radius * (-np.sin(theta) + 1j * np.cos(theta)))
    F = lambda theta: np.abs(f(a2c(theta)))
    H_w_z = lambda theta: bound_h_w_z(w, a2c(theta), S[0], S[1])
    integrand = lambda theta: F(theta) * dr(theta)  # * H_w_z(theta) ** (m + 1)  # this is 1 anyway
    integral, abserr = scipy.integrate.quad(integrand, 0, 2 * np.pi)
    if abserr > integral:
        print(f"integral: {integral}, abserr: {abserr}")
    result = integral / 2 / np.pi * cg_bound
    return m, result


def get_cg_bound(kappa, m):
    """The well known kappa bound for the CG error."""
    return 2 * ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** m


def get_restarted_cg_bound(kappa, m, starts=1):
    """Iterate the kappa CG bound."""
    bound = get_cg_bound(kappa, m)
    return bound ** np.arange(1, starts + 1)


def get_cg_errors(A, w, b, n, solution=None, norm=scipy.linalg.norm):
    """
    Get the real 2-norm errors of CG applied to (A-wI)x=b.
    This is not realistic as part of an a-priori error indicator but for theoretical purposes.
    """
    if solution is None:
        solution = scipy.sparse.linalg.spsolve(A - w * scipy.sparse.eye(*A.shape), b)
    errors = [norm(solution)]

    def callback(xk):
        error = norm(solution - xk)
        errors.append(error)

    scipy.sparse.linalg.cg(A - w * scipy.sparse.eye(*A.shape), b, x0=np.zeros_like(b), maxiter=n, callback=callback)
    cg_errors = np.zeros(n + 1)
    cg_errors[:len(errors)] = errors
    return cg_errors


def get_restarted_cg_errors(A, w, b, m, starts=1, solution=None, norm=scipy.linalg.norm):
    """
    Get the 2-norm errors of restarted CG applied to (A-wI)x=b.
    This is not realistic as part of an a-priori error indicator but for theoretical purposes.
    """
    errors = []
    if solution is None:
        solution = scipy.sparse.linalg.spsolve(A - w * scipy.sparse.eye(*A.shape), b)
    errors = [norm(solution)]
    x0 = np.zeros_like(solution)
    for _ in range(starts):
        x0, _ = scipy.sparse.linalg.cg(A - w * scipy.sparse.eye(*A.shape), b, x0=x0, maxiter=m)
        errors.append(norm(solution - x0))
    cg_errors = np.zeros(starts + 1)
    cg_errors[:len(errors)] = errors
    return cg_errors


def h_w_z(x, w, z):
    return (x - w) / (x - z)


def bound_h_w_z(w, z, a, b):
    """Bound the expression (x - w) / (x - z) on the interval [a,b]."""
    result = np.zeros_like(z)
    x_star = (np.real(z) ** 2 + np.imag(z) ** 2 - np.real(z) * w)
    np.divide((z - w), np.imag(z), out=result,
              where=(a * (np.real(z) - w) <= x_star) * (x_star <= b * (np.real(z) - w)))
    # result = np.where((a * (np.real(z) - w) <= x_star) * (x_star <= b * (np.real(z) - w)),
    #                   np.abs((z - w) / np.imag(z)), 0)
    result = np.maximum(np.maximum(np.abs((a - w) / (a - z)), np.abs((b - w) / (b - z))), np.abs(result))
    result = np.where(np.equal(w, z) * np.equal(z, 0), 1, result)
    return result


def get_determinant_tridiag(T, shift=0.0, return_all=False):
    n = T.shape[0]
    assert n > 1
    # f3 = 0  # f_{n-3}
    f = [1]
    f.append(T[0, 0] - shift)
    for i in range(1, n):
        a, b, c = T[i, i] - shift, T[i - 1, i], T[i, i - 1]
        fi = a * f[-1] - b * c * f[-2]
        f.append(fi)
    if not return_all:
        return f[-1]
    else:
        return np.array(f)


def chen_musco_post(T: np.array, w: float, f=np.exp, fix_0_eval=True):
    """
    A posteriori error bound for Lanczos approximation of f(A).
    In comparison to the a priori bound the tridiagonal matrix T is available.

    Source: Section 3.2 T. Chen, A. Greenbaum, C. Musco, and C. Musco,
    “Error Bounds for Lanczos-Based Matrix Function Approximation,”
    SIAM J. Matrix Anal. Appl., vol. 43, no. 2, pp. 787–811, Jun. 2022, doi: 10.1137/21M1427784.
    """
    ritz = np.sort(scipy.linalg.eigvalsh(T))
    sm_eval = ritz[0]
    la_eval = ritz[-1]
    c, r, kappa = setup_circle_and_kappa(sm_eval, la_eval, w, fix_0_eval=fix_0_eval)

    a2c = lambda theta: c + r * (np.cos(theta) + 1j * np.sin(theta))
    dr = lambda theta: np.abs(r * (-np.sin(theta) + 1j * np.cos(theta)))
    F = lambda theta: np.abs(f(a2c(theta)))
    H_w_z = lambda theta: bound_h_w_z(w, a2c(theta), sm_eval, la_eval)
    Dets = lambda theta: np.abs(get_determinant_tridiag(T, w, True) / get_determinant_tridiag(T, a2c(theta), True))
    integrand = lambda theta: F(theta) * H_w_z(theta) * Dets(theta) * dr(theta)
    integral, abserr = scipy.integrate.quad_vec(integrand, 0, 2 * np.pi)
    if np.any(abserr > integral):
        print(f"integral: {integral}, abserr: {abserr}")

    m = np.arange(T.shape[0] + 1)
    cg_bound = get_cg_bound(kappa, m)
    return m, integral / 2 / np.pi * cg_bound


def chen_musco_post_no_kappa(A, b, T: np.array, w: float, center: float, radius: float, f=np.exp):
    """
    A posteriori error bound for Lanczos approximation of f(A).
    In comparison to the a priori bound the tridiagonal matrix T is available.

    Source: Section 3.2 T. Chen, A. Greenbaum, C. Musco, and C. Musco,
    “Error Bounds for Lanczos-Based Matrix Function Approximation,”
    SIAM J. Matrix Anal. Appl., vol. 43, no. 2, pp. 787–811, Jun. 2022, doi: 10.1137/21M1427784.
    """
    ritz = np.sort(scipy.linalg.eigvalsh_tridiagonal(np.diag(T), np.diag(T, -1)))
    sm_eval = ritz[0]
    la_eval = ritz[-1]
    assert w < sm_eval or w > la_eval

    a2c = lambda theta: center + radius * (np.cos(theta) + 1j * np.sin(theta))
    dr = lambda theta: np.abs(radius * (-np.sin(theta) + 1j * np.cos(theta)))
    F = lambda theta: np.abs(f(a2c(theta)))
    H_w_z = lambda theta: bound_h_w_z(w, a2c(theta), sm_eval, la_eval)
    Dets = lambda theta: np.abs(get_determinant_tridiag(T, w, True) / get_determinant_tridiag(T, a2c(theta), True))
    integrand = lambda theta: F(theta) * H_w_z(theta) * Dets(theta) * dr(theta)
    integral, abserr = scipy.integrate.quad_vec(integrand, 0, 2 * np.pi)
    if np.any(abserr > integral):
        print(f"integral: {integral}, abserr: {abserr}")

    m = T.shape[0]
    cg_bound = get_cg_errors(A, w, b, m + 1)
    return np.arange(m + 1), integral / 2 / np.pi * cg_bound


def restarted_post_no_kappa(A, b, T_small: np.array, w: float, center: float, radius: float, starts=1, f=np.exp):
    """Variation of the Chen et al. bound above but for restarted Lanczos."""
    ritz = np.sort(scipy.linalg.eigvals(T_small))
    sm_eval = ritz[0]
    la_eval = ritz[-1]
    assert w < sm_eval or w > la_eval

    rs = np.arange(1, starts + 1)

    a2c = lambda theta: center + radius * (np.cos(theta) + 1j * np.sin(theta))
    dr = lambda theta: np.abs(radius * (-np.sin(theta) + 1j * np.cos(theta)))
    F = lambda theta: np.abs(f(a2c(theta)))
    H_w_z = lambda theta: bound_h_w_z(w, a2c(theta), sm_eval, la_eval)
    Dets = lambda theta: np.abs(
        get_determinant_tridiag(T_small, w) / get_determinant_tridiag(T_small, a2c(theta))) ** rs
    integrand = lambda theta: F(theta) * H_w_z(theta) * Dets(theta) * dr(theta)
    integral, abserr = scipy.integrate.quad_vec(integrand, 0, 2 * np.pi)
    if np.any(abserr > integral):
        print(f"integral: {integral}, abserr: {abserr}")

    m = T_small.shape[0]
    cg_bound = get_restarted_cg_errors(A, w, b, m, starts=starts)
    return np.arange(m, (starts + 1) * m, m), integral / 2 / np.pi * cg_bound


def restarted_post(T_small: np.array, w: float, starts=1, f=np.exp, fix_0_eval=True):
    """
    Variation of the Chen et al. bound for restarted Lanczos.
    """
    ritz = np.sort(scipy.linalg.eigvals(T_small))
    sm_eval = ritz[0]
    la_eval = ritz[-1]
    c, r, kappa = setup_circle_and_kappa(sm_eval, la_eval, w, fix_0_eval=fix_0_eval)

    starts_ra = np.arange(1, starts + 1)

    a2c = lambda theta: c + r * (np.cos(theta) + 1j * np.sin(theta))
    dr = lambda theta: np.abs(r * (-np.sin(theta) + 1j * np.cos(theta)))
    F = lambda theta: np.abs(f(a2c(theta)))
    H_w_z = lambda theta: bound_h_w_z(w, a2c(theta), sm_eval, la_eval)
    Dets = lambda theta: np.abs(
        get_determinant_tridiag(T_small, w) / get_determinant_tridiag(T_small, a2c(theta))) ** starts_ra
    integrand = lambda theta: F(theta) * H_w_z(theta) * Dets(theta) * dr(theta)
    integral, abserr = scipy.integrate.quad_vec(integrand, 0, 2 * np.pi)
    if np.any(abserr > integral):
        print(f"integral: {integral}, abserr: {abserr}")

    m = T_small.shape[0]
    cg_bound = get_restarted_cg_bound(kappa, m, starts)
    return np.arange(m, (starts + 1) * m, m), integral / 2 / np.pi * cg_bound


def afanasjew_post(T: np.array, v: np.array, A: np.array, m: int, f: callable):
    """
    Error indicator for the restarted Arnoldi method for f(A)b.

    Source: Section 4 M. Afanasjew, M. Eiermann, O. G. Ernst, and S. Güttel,
    “Implementation of a restarted Krylov subspace method for the evaluation of matrix functions,”
    Linear Algebra and its Applications, vol. 429, no. 10, pp. 2293–2314, Nov. 2008, doi: 10.1016/j.laa.2008.06.029.
    """
    assert m < T.shape[0]
    delta = T[m + 1, m]
    Tm = T[:m, :m]
    ritz = np.real(np.sort(scipy.linalg.eigvals(Tm)))
    sm_eval = ritz[0]
    la_eval = ritz[-1]
    B = np.array([[sm_eval, 0], [1, la_eval]])
    E = np.zeros((2, Tm.shape[1]))
    E[0, -1] = delta
    H_tilde = np.block([[Tm, np.zeros((Tm.shape[0], 2))], [E, B]])
    fH = f(H_tilde)
    phi_1 = np.abs(fH[-2, 0])
    phi_2 = np.abs(fH[-1, 0])
    lower = phi_1 * v
    upper = lower + phi_2 * (A @ v - la_eval * v)
    return np.array([np.linalg.norm(lower), np.linalg.norm(upper)])


def afanasjew_post_for_plot(T, v, A, m, starts, f):
    bounds = np.zeros((2, starts))
    for i in range(starts):
        bounds[:, i] = afanasjew_post(T, v, A, (i + 1) * m, f)
    return np.arange(m, (starts + 1) * m, m), bounds


def saad_post(H: np.array, m: int, f: callable):
    """
    Error indicator for the restarted Arnoldi method for exp(A)b.

    Source: Section 5 Y. Saad, “Analysis of Some Krylov Subspace Approximations to the Matrix Exponential Operator,”
    SIAM J. Numer. Anal., vol. 29, no. 1, pp. 209–228, Feb. 1992, doi: 10.1137/0729014.

    """
    assert m < H.shape[0]
    delta = H[m + 1, m]
    Hm = H[:m, :m]
    lower_left = np.zeros((1, Hm.shape[1]))
    lower_left[0, -1] = delta
    H_tilde = np.block([[Hm, np.zeros((Hm.shape[0], 1))], [lower_left, np.zeros((1, 1))]])  # Assuming beta = 1
    fH = f(H_tilde)
    phi_1 = np.abs(fH[-1, 0])
    return phi_1


def saad_post_for_plot(H, m, starts, f):
    bounds = []
    for i in range(starts):
        bounds.append(saad_post(H, (i + 1) * m, f))
    return np.arange(m, (starts + 1) * m, m), bounds
