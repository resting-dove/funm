import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

from src.matfuncb.krylov_basis import arnoldi
from src.matfuncb.matfuncb import matfuncb
from test_np.utils.matrix_factory import get_symmetric_matrix_by_evals


# Investigate how the error bounds for the Lanczos method look for the rough matrix we have in MD

def hochbruck_lubich(sm_ev: float, t: float, n: int):
    """Not sure which norm this is."""
    assert sm_ev < 0
    rho = sm_ev / -4

    start = min(int(np.sqrt(4 * rho * t)), n - 1)
    end = min(int(2 * rho * t), n)
    m = np.arange(start, end)
    ret = 10 * np.exp(- m ** 2 / (5 * rho * t))
    m = np.arange(end, n)
    ret2 = 10 / (rho * t) * np.exp(-rho * t) * (np.e * rho * t / m) ** m
    return np.arange(start, n), np.array(list(ret) + list(ret2))


def saad(norm_A: float, t: float, n: int):
    """Output is in norm compatible with input norm.
    As reported by Ye and copied by me."""
    assert t * norm_A <= 1
    m = np.arange(1, n)
    return m, 2 / scipy.special.factorial(m) * (t * norm_A / 2) ** m


def ye_entry_1(m: np.array, t: float, sm_eval: float, q=0.4):
    assert sm_eval < 0
    b = -sm_eval
    a = 0
    q0 = (np.sqrt(b) - np.sqrt(a)) / (np.sqrt(b) + np.sqrt(a))
    gamma = (b - a) * (q - q0) * (1 / q0 - q)
    return 2 / (1 - q) * np.exp(-t * gamma / 4 / q) * q ** (m - 1)


def ye_entry_2(m: np.array, t: float, sm_eval: float):
    assert sm_eval < 0
    b = -sm_eval
    return 1 / (scipy.special.factorial(m - 1)) * (t * b / 2) ** (m - 1)


def ye(sm_eval: float, t: float, n: int, alpha: float):
    """"""
    assert alpha <= t
    m = np.arange(1, n)
    beta = 1  # This could be anything really but doesn't change shape
    lambda_min = 0
    h0 = ye_entry_1(m, 0, sm_eval)
    hhalf = ye_entry_1(m, alpha / 2, sm_eval)
    halpha = ye_entry_1(m, alpha, sm_eval)
    hthreequart = ye_entry_2(m, alpha + (t - alpha) / 2, sm_eval)
    hthreequart2 = ye_entry_2(m, alpha + (t - alpha) / 2, sm_eval)
    ht = ye_entry_1(m, t, sm_eval)
    ht2 = ye_entry_2(m, t, sm_eval)
    left = np.maximum(h0, hhalf, halpha)
    right = np.minimum(np.maximum(halpha, hthreequart, ht), np.maximum(hthreequart2, ht2))
    # result = beta * (h0alpha * np.exp((alpha - t) * lambda_min) * alpha + ye_entry_2(m, t, sm_eval) * (t - alpha))
    result = beta * (left * np.exp((alpha - t) * lambda_min) * alpha + right * (t - alpha))
    return m, result


def get_points_on_circle(c: float, r: float, n: int):
    theta = np.linspace(0, 2 * np.pi, n)
    return c + r * (np.cos(theta) + 1j * np.sin(theta))


def chen_musco(sm_eval: float, w: float, n: int, f=np.exp):
    """Basically shift to the system (A-wI) and use an error bound for the k-th CG step of (A-wI)x=b."""
    assert sm_eval < 0
    assert w > 0
    la_eval = 0
    kappa = (sm_eval - w) / (la_eval - w)
    sqrt_kappa = np.sqrt(kappa)
    m = np.arange(1, n)
    # cg_bound = 2 * ((sqrt_kappa - 1) / (sqrt_kappa + 1)) ** m
    cg_bound = 2 * np.exp(- 2 * m / sqrt_kappa)
    result = np.abs(sm_eval - w) * np.max(np.abs(f(get_points_on_circle(sm_eval, sm_eval - w, 20)))) * cg_bound
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


def chen_musco_post(T: np.array, w: float, f=np.exp, ):
    """Posteriori bound for arbitrary f."""
    ritz = np.sort(scipy.linalg.eigvals(T))
    sm_eval = ritz[0]
    la_eval = 0  # ritz[-1]
    assert w < sm_eval or w > la_eval
    eps = 10
    c, r = w, max(np.abs(w - sm_eval), np.abs(w - la_eval)) + eps
    a2c = lambda theta: c + r * (np.cos(theta) + 1j * np.sin(theta))
    dr = lambda theta: np.abs(r * (-np.sin(theta) + 1j * np.cos(theta)))
    F = lambda theta: np.abs(f(a2c(theta)))
    H_w_z = lambda theta: bound_h_w_z(w, a2c(theta), sm_eval, la_eval)
    Dets = lambda theta: np.abs(get_determinant_tridiag(T, w) / get_determinant_tridiag(T, a2c(theta)))
    integrand = lambda theta: F(theta) * H_w_z(theta) * Dets(theta) * dr(theta)
    integral, abserr = scipy.integrate.quad(integrand, 0, 2 * np.pi, epsabs=10)
    if abserr > integral:
        print(f"integral: {integral}, abserr: {abserr}")
    kappa = (sm_eval - w) / (la_eval - w)
    if 0 < kappa < 1:  # (T -wI) has flipped sign compared to T and thus the eigvals flip.
        kappa = 1 / kappa
    sqrt_kappa = np.sqrt(kappa)
    m = T.shape[0]
    # cg_bound = 2 * ((sqrt_kappa - 1) / (sqrt_kappa + 1)) ** m
    cg_bound = 2 * np.exp(- 2 * m / sqrt_kappa)
    return m, integral * cg_bound


def expm(A):
    return scipy.linalg.expm(A.toarray())


if __name__ == "__main__":
    evals = -1 * np.array(
        [2000, 20, 20, 20, 20] + list(np.arange(10, 1000, 100)) + list(np.arange(0.1, 10, 0.01)) + [0, 0, 0, 0, 0, 0])
    n = min(len(evals), 100)
    Omega2, S = get_symmetric_matrix_by_evals(evals, True, "./precalculated", load=True, save=False)
    t = 1 / 21  # seconds
    b = np.random.random(len(evals))

    exact = S.T @ np.diag(np.exp(t * evals)) @ S @ b
    max_acc = 1e-16
    errors = []
    i_s = []
    for i in range(2, 100):  # int(len(evals) / 2)):
        app, info = matfuncb(t * Omega2, b, expm, k=i, symmetric=True)
        err = np.linalg.norm(app - exact)
        errors.append(err)
        i_s.append(i)
        if err <= max_acc:
            break
    plt.plot(i_s, errors, "x")
    plt.title(r"A priori error bounds for $\exp(-tA)$ with $||A||_2 = 2000$.")
    plt.plot(*hochbruck_lubich(evals[0], t, n), label="Hochb&Lub", linestyle="solid")
    if np.abs(evals[0]) * t <= 1:
        plt.plot(*saad(np.abs(evals[0]), t, n), label="Saad", linestyle="dashed")
    plt.plot(*ye(evals[0], t=t, n=n, alpha=0), label="Ye 0", linestyle="dotted")
    plt.plot(*ye(evals[0], t=t, n=n, alpha=0.5 * t), label="Ye 0.5t", linestyle="dotted")
    plt.plot(*ye(evals[0], t=t, n=n, alpha=t), label="Ye t", linestyle="dotted")
    plt.plot(*chen_musco(t * evals[0], w=1, n=n), label="Musco", linestyle="dashdot")
    plt.yscale("log")
    # plt.ylim(top=100, bottom=1e-24)
    plt.legend()
    plt.show()
    w = b / np.linalg.norm(b)
    (w, V, T, breakdown) = arnoldi(t * Omega2, w, 40)
    plt.plot(i_s, errors, "x")
    plt.title(r"A posteriori error bounds for $\exp(-tA)$ with $||A||_2 = 2000$.")
    # plt.scatter(*chen_musco_post(T[:3, :3], w=-100), label="Musco")
    plt.plot(*chen_musco_post(T[:10, :10], w=-100), "or", label="Musco")
    plt.plot(*chen_musco_post(T[:25, :25], w=-100), "or", label="Musco")
    plt.plot(*chen_musco_post(T[:40, :40], w=-100), "or", label="Musco")
    plt.yscale("log")
    plt.legend()
    plt.show()

    plt.plot(np.arange(50), ye_entry_1(np.arange(50), t, evals[0], q=0.4), label="q=0.4")
    plt.plot(np.arange(1000), ye_entry_1(np.arange(1000), t, evals[0], q=0.1), label="q=0.1")
    plt.plot(np.arange(1000), ye_entry_1(np.arange(1000), t, evals[0], q=0.5), label="q=0.5")
    plt.plot(np.arange(1000), ye_entry_1(np.arange(1000), t, evals[0], q=0.9), label="q=0.9")
    plt.plot(np.arange(1000), ye_entry_2(np.arange(1000), t, evals[0]), label="Second")
    plt.yscale("log")
    # plt.ylim(top=1000, bottom=1e-32)
    plt.legend()
    plt.show()
    1 + 1
