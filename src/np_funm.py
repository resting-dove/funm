import numpy as np
from Arnoldi import extend_arnoldi
import scipy


def funm_krylov(A, b: np.array, param):
    n = b.shape[0]
    beta = np.linalg.norm(b)
    w = b
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
