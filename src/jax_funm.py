import jax
from jax import numpy as jnp
from jax_Arnoldi import extend_arnoldi
import scipy


def funm_krylov(A, b: jax.Array, param):
    n = b.shape[0]
    beta = jnp.linalg.norm(b)
    w = b
    m = param["restart_length"]
    V_big = jnp.zeros((n, param["num_restarts"] * m + 20), b.dtype)
    f = jnp.zeros_like(b)
    H_full = jnp.zeros((m * param["num_restarts"] + 1, m * param["num_restarts"]), dtype=b.dtype)
    fs = jnp.zeros((n, param["num_restarts"]))
    eigvals = {}
    update_norms = []
    for k in range(param["num_restarts"]):
        H = jnp.zeros((m + 1, m + 1))
        V_big = V_big.at[:, k * m].set(w)

        (w, V_big, H, h, breakdown) = extend_arnoldi(A=A, V_big=V_big, H=H, s=k * m, m=(k + 1) * m)
        # (w, V_big, H, h, breakdown) = (
        #    jit(Arnoldi_2, static_argnames=["steps", "trunc", "reorth_num"])(A, V_big, H, s=k * m, steps=m, trunc=m))
        H_full = H_full.at[k * m: (k + 1) * m, k * m: (k + 1) * m].set(H)
        # if k > 0:
        H_full = H_full.at[(k + 1) * m, (k + 1) * m - 1].set(h)

        H_exp = scipy.linalg.expm(H_full[: (k + 1) * m, : (k + 1) * m])
        H_exp_jax = jnp.array(H_exp)[-m:, 0]
        f = beta * (V_big[:, k * m: (k + 1) * m] @ H_exp_jax) + f
        fs = fs.at[:, k].set(f)
        eigvals[k] = jnp.linalg.eigvals(H_full[:(k + 1) * m, :(k + 1) * m])
        update_norms.append(jnp.linalg.norm(beta * H_exp_jax))
    return fs, eigvals, update_norms
