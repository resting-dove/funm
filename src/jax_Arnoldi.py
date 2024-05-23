import jax
from jax import numpy as jnp


def extend_arnoldi(A, V_big: jax.Array, m: int, H: jax.Array, s: int, trunc=-1, reorth_num=0):
    """Extend a given Arnoldi decomposition (V_big, H) of dimension s up to dimention m.
    Assume that H is at least of size (m+1, m+1)."""
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
