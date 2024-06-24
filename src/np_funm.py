import numpy as np
from src.Arnoldi import extend_arnoldi, arnoldi
from src.gershgorin import gershgorin
from src.power_method import power_method
import scipy


def funm_krylov(A, b: np.array, param):
    n = b.shape[0]
    beta = np.linalg.norm(b)
    w = b / beta
    m = param["restart_length"]
    V_big = np.zeros((n, param["num_restarts"] * m + 20))
    f = np.zeros_like(b)
    H_full = np.zeros((m * param["num_restarts"] + 1, m * param["num_restarts"]))
    fs = np.zeros((n, param["num_restarts"]))
    eigvals = {}
    update_norms = []
    for k in range(param["num_restarts"]):
        V_big[:, k * m] = w

        (w, V_big, H, h, breakdown) = extend_arnoldi(A=A, V_big=V_big, s=k * m, m=(k + 1) * m)
        # (w, V_big, H, h, breakdown) = (
        #    jit(Arnoldi_2, static_argnames=["steps", "trunc", "reorth_num"])(A, V_big, H, s=k * m, steps=m, trunc=m))
        H_full[k * m: (k + 1) * m, k * m: (k + 1) * m] = H
        # if k > 0:
        H_full[(k + 1) * m, (k + 1) * m - 1] = h

        H_exp = scipy.linalg.expm(H_full[: (k + 1) * m, : (k + 1) * m])
        H_exp_jax = np.array(H_exp)[-m:, 0]
        f = beta * (V_big[:, k * m: (k + 1) * m] @ H_exp_jax) + f
        fs[:, k] = f
        eigvals[k] = np.linalg.eigvals(H_full[:(k + 1) * m, :(k + 1) * m])
        update_norms.append(np.linalg.norm(beta * (V_big[:, k * m: (k + 1) * m] @ H_exp_jax)))
    return fs, eigvals, update_norms

def funm_krylov_v2(A, b: np.array, param, calculate_eigvals=True, stopping_acc=1e-10):
    """Variation on the restarted Krylov implementation, influenced by the constraints that Jax puts on variable
    shapes."""
    stopping_criterion = False
    n = b.shape[0]
    beta = float(np.linalg.norm(b))
    w = b / beta
    m = param["restart_length"]
    f = np.zeros_like(b)
    H_full = np.zeros((m * param["num_restarts"] + 2, m * param["num_restarts"]), dtype=b.dtype)
    fs = np.zeros((n, param["num_restarts"]))
    eigvals = {}
    update_norms = []
    for k in range(param["num_restarts"]):
        if stopping_criterion:
            break
        (w, V, H, breakdown) = arnoldi(A=A, w=w, m=m)
        if breakdown:
            print("breakdown")
            stopping_criterion = True
        H_full[k * m: (k + 1) * m + 1, k * m: (k + 1) * m] = H
        H_exp = scipy.linalg.expm(H_full[: (k + 1) * m, : (k + 1) * m])
        H_exp_jax = np.array(H_exp)[-m:, 0]
        f = beta * (V @ H_exp_jax) + f
        fs[:, k] = f
        update = np.linalg.norm(beta * H_exp_jax)
        if calculate_eigvals:
            eigvals[k] = np.linalg.eigvals(H_full[:(k + 1) * m, :(k + 1) * m])
        update_norms.append(update)
        if update / np.linalg.norm(f) < stopping_acc:
            stopping_criterion = True
            print("Stopping accuracy reached.")
        if k > 10 and (update / update_norms[-1] < .05):
            stopping_criterion = True
            print("Updates getting to small.")

    return fs, eigvals, update_norms


def gershgorin_adaptive_expm(A, b: np.array, calculate_eigvals=True, stopping_acc=1e-10):
    """Evaluation of exp(A)b using an adaptive krylov size.
    For now not restarted"""
    param = {"num_restarts": 1}
    low, high = gershgorin(-A)
    print("For now assume the spectrum of A is in [-high, 0].")
    rho = high / 4
    # Use the first error bound from the paper
    m = int(np.sqrt(-np.log(stopping_acc / 10) * 5 * rho))
    assert np.sqrt(4 * rho) <= m
    assert m <= 2 * rho
    print(f"m is set to {m}.")
    param["restart_length"] = m
    fs, eigvals, update_norms = funm_krylov_v2(A, b, param, calculate_eigvals, stopping_acc)
    return fs, eigvals, update_norms, m


def power_adaptive_expm(A, b: np.array, calculate_eigvals=True, stopping_acc=1e-10):
    """Evaluation of exp(A)b using an adaptive krylov size derived from a few ppower iteration steps.
    For now not restarted"""
    param = {"num_restarts": 1}
    high = power_method(-A, b, 3)
    print("For now assume the spectrum of A is in [-high, 0].")
    rho = high / 4
    # Use the first error bound from the paper
    m = int(np.sqrt(-np.log(stopping_acc / 10) * 5 * rho))
    assert np.sqrt(4 * rho) <= m
    assert m <= 2 * rho
    print(f"m is set to {m}.")
    param["restart_length"] = m
    fs, eigvals, update_norms = funm_krylov_v2(A, b, param, calculate_eigvals, stopping_acc)
    return fs, eigvals, update_norms, m