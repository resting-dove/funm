import jax
from jax import numpy as jnp
import numpy as np
from functools import partial


def extend_arnoldi(A, V_big: jax.Array, m: int, s: int, trunc=-1, reorth_num=0):
    """Extend a given Arnoldi decomposition of dimension s up to dimention m.
    """
    H = jnp.zeros((m + 1, m + 1))
    eps = 1e-15
    trunc = trunc if trunc >= 0 else m - s
    breakdown = False
    w = V_big[:, s]

    # make the k column in H_full and the k+1 column in V
    # this is the k - s column in H
    for k in jnp.arange(s, m):
        k_small = k - s
        w = V_big[:, k]
        w = jnp.dot(A, w)

        sj = max([s, k - trunc])  # start orthogonalizing from this index
        for j in jnp.arange(sj, k + 1):
            v = V_big[:, j]
            ip = jnp.dot(v, w)
            H = H.at[j - s, k_small].add(ip)
            w = w - ip * v
        w2 = jnp.dot(w, w)
        H = H.at[k_small + 1, k_small].set(jnp.sqrt(w2))

        if H[k_small + 1, k_small] < k * eps:
            breakdown = True
            print("breakdown")

        w = w / H[k_small + 1, k_small]
        if k < m:
            V_big = V_big.at[:, k + 1].set(w)

    h = H[m - s, m - s - 1]
    H = H[:m - s, :m - s]
    return w, V_big, H, h, breakdown


@partial(jax.jit, static_argnames=["m"])
def arnoldi_jittable(A, w: jax.Array, m: int):
    """Calculate an Arnoldi decomposition of dimension m.
    V_big might be an earlier the basis from earlier Arnoldi decompositions.
    """
    H = jnp.zeros((m + 1, m + 1))
    new_V_big = jnp.empty((w.shape[0], m))

    trunc = m
    new_V_big = new_V_big.at[:, 0].set(w)
    # make the k column in H_full and the k+1 column in V
    # this is the k - s column in H
    for k_small in np.arange(m):
        w = new_V_big[:, k_small]
        w = jnp.dot(A, w)

        sj = 0  # jax.lax.max(0, k_small - trunc)  # start orthogonalizing from this index
        for j in jnp.arange(sj, k_small + 1):
            v = new_V_big[:, j]
            ip = jnp.dot(v, w)
            H = H.at[j, k_small].add(ip)
            w = w - ip * v
        w2 = jnp.dot(w, w)
        H = H.at[k_small + 1, k_small].set(jnp.sqrt(w2))

        w = w / H[k_small + 1, k_small]
        if k_small < m - 1:
            new_V_big = new_V_big.at[:, k_small + 1].set(w)

    #h = H[m, m - 1]
    H = H[:m + 1, :m]

    return w, new_V_big, H#, h
