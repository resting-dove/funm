from functools import partial

import jax
import scipy
from jax import numpy as jnp
import numpy as np

from jax_Arnoldi import extend_arnoldi, arnoldi_jittable


def funm_krylov(A, b: jax.Array, param):
    n = b.shape[0]
    beta = jnp.linalg.norm(b)
    w = b / beta
    m = param["restart_length"]
    V_big = jnp.zeros((n, param["num_restarts"] * m + 20))
    f = jnp.zeros_like(b)
    H_full = jnp.zeros((m * param["num_restarts"] + 1, m * param["num_restarts"]))
    fs = jnp.zeros((n, param["num_restarts"]))
    update_norms = []
    for k in range(param["num_restarts"]):
        V_big = V_big.at[:, k * m].set(w)

        (w, V_big, H, h, breakdown) = extend_arnoldi(A=A, V_big=V_big, s=k * m, m=(k + 1) * m)
        # (w, V_big, H, h, breakdown) = (
        #    jit(Arnoldi_2, static_argnames=["steps", "trunc", "reorth_num"])(A, V_big, H, s=k * m, steps=m, trunc=m))
        H_full = H_full.at[k * m: (k + 1) * m, k * m: (k + 1) * m].set(H)
        # if k > 0:
        H_full = H_full.at[(k + 1) * m, (k + 1) * m - 1].set(h)

        H_exp = scipy.linalg.expm(H_full[: (k + 1) * m, : (k + 1) * m])
        H_exp_jax = jnp.array(H_exp)[-m:, 0]
        f = beta * (V_big[:, k * m: (k + 1) * m] @ H_exp_jax) + f
        fs = fs.at[:, k].set(f)
        update_norms.append(jnp.linalg.norm(beta * H_exp_jax))

    return fs, update_norms


@partial(jax.jit, static_argnames=["m", ])
def calculate_loop_p1(A, w, m, H_full, k):
    (w, V, H, breakdown) = arnoldi_jittable(A=A, w=w, m=m)
    #V_big = jnp.concat([V_big, new_V_big], axis=1)
    H_full = jax.lax.dynamic_update_slice(H_full, H, (k * m, k * m))
    return V, w, H_full


@partial(jax.jit, static_argnames=["beta"])
def calculate_loop_p2(H_exp, beta, V, f):
    f = beta * (V @ H_exp) + f
    return f


# @partial(jax.jit, static_argnames=["m", "beta"], device=jax.devices('cpu')[0])
def calculate_loop(A: jax.Array, H_full: jax.Array, f: jax.Array, w: jax.Array, m: int, k: int,
                   beta: float):
    V, w, H_full = calculate_loop_p1(A, w, m, H_full, k)

    # H_exp = jax.scipy.linalg.expm(H_full[: (k + 1) * m, : (k + 1) * m])
    H_slice = jax.lax.dynamic_slice(H_full, (0, 0), ((k + 1) * m, (k + 1) * m))
    H_exp = jax.scipy.linalg.expm(H_slice)[-m:, 0]
    f = calculate_loop_p2(H_exp, beta, V, f)
    return f, H_full, w


def funm_krylov_jittable(A, b: jax.Array, param, tol=1e-5):
    cpu_device = jax.devices('cpu')[0]
    n = b.shape[0]
    beta = float(jnp.linalg.norm(b))
    w = b / beta
    m = param["restart_length"]
    f = jnp.zeros_like(b)
    H_full = jnp.zeros((m * param["num_restarts"] + 1, m * param["num_restarts"]))
    fs = jnp.zeros((n, param["num_restarts"]))
    update_norms = []
    updates = []
    stopping_criterion = False
    prev = f
    for k in range(param["num_restarts"]):
        if stopping_criterion:
            break
        f, H_full, w = calculate_loop(A, H_full, f, w, m, k, beta)
        fs = fs.at[:, k].set(f)
        updates.append(np.linalg.norm(f - prev))
        prev = f
        if k > 3 and updates[-1] < tol:
            stopping_criterion = True
            print("Updates getting too small")


    return fs  # , update_norms
