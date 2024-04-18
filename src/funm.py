import jax
import numpy as np
import jax.numpy as jnp
from jax import random, jit, vmap
import scipy
import matplotlib.pyplot as plt


def get_orthogonal_vector(V: jax.Array, w: jax.Array):
    """Orthogonalize the vector w against all columns in V."""
    for i in range(V.shape[1]):
        v = V[:, i]
        update = jnp.dot(v, w)
        w = w - update * v
    return w


def get_canonical_vector(n: int, k: int):
    """Get the k-th canonical basis vector of length n."""
    x = jnp.zeros(n)
    x = x.at[k].add(1)
    return x


def orthogonalize(M: jnp.array, n: int, trunc=jnp.inf):
    for j in jnp.arange(n, dtype=int):
        w = M[:, j]
        sj = jnp.clip(j - trunc, 0)
        for k in jnp.arange(sj, j, dtype=int):
            v = M[:, k]
            ip = jnp.dot(w, v)
            w = w - ip * v
            w = w / jnp.linalg.norm(w)
        w = w / jnp.linalg.norm(w)
        M = M.at[:, j].set(w)
    return M


def Arnoldi(A, V_big: jax.Array, m: int, H: jax.Array, s: int, trunc=-1, reorth_num=0):
    """Extend a given Arnoldi decomposition (V_big, H) of dimension s up to dimention m.
    Assume that H is at least of size (m+1, m+1)."""
    eps = 1e-15
    reo = reorth_num
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

        H = H.at[k_small + 1, k_small].set(jnp.sqrt(jnp.dot(w, w)))

        if H[k_small + 1, k_small] < k * eps:
            breakdown = True
            print("breakdown")

        w = w / H[k_small + 1, k_small]
        # if k < m - 1:
        V_big = V_big.at[:, k + 1].set(w)

    h = H[m - s, m - s - 1]
    H = H[:m - s, :m - s]
    return w, V_big, H, h, breakdown


def funm_krylov(A, b: jax.Array, param):
    should_stop = False

    n = b.shape[0]
    beta = jnp.linalg.norm(b)
    v = b / beta
    m = param["restart_length"]
    V_big = jnp.zeros((n, param["num_restarts"] * m + 20), b.dtype)
    f = jnp.zeros_like(b)
    H_full = jnp.zeros((m * param["num_restarts"], m * param["num_restarts"]), dtype=b.dtype)
    fs = jnp.zeros((n, param["num_restarts"]))
    for k in range(param["num_restarts"]):
        if should_stop:
            break
        H = jnp.zeros((m + 1, m + 1))
        V_big = V_big.at[:, k * m].set(v)

        (w, V_big, H, h, breakdown) = Arnoldi(A, V_big, (k + 1) * m, H, k * m)
        H_full = H_full.at[k * m: (k + 1) * m, k * m: (k + 1) * m].set(H)
        # if k > 0:
        H_full = H_full.at[(k + 1) * m, (k + 1) * m - 1].set(h)

        H_exp = scipy.linalg.expm(H_full[: (k + 1) * m, : (k + 1) * m])
        H_exp_jax = jnp.array(H_exp)
        f = beta * (V_big[:, k * m: (k + 1) * m] @ H_exp_jax[-m:, 0]) + f
        fs = fs.at[:, k].set(f)
    return fs


if __name__ == "__main__":
    n = 200
    EWs = jnp.arange(1, n + 1) / n
    key = random.key(0)
    key, *subkeys = random.split(key)
    initializer = jax.nn.initializers.orthogonal()
    # S = initializer(key, (n, n))
    # S = jnp.eye(n, n)
    S = random.uniform(subkeys[0], (n, n))
    S = orthogonalize(S, n)
    A = S @ jnp.diag(EWs) @ S.transpose()
    b = jnp.ones(n) / jnp.linalg.norm(jnp.ones(n))

    param = {
        "restart_length": 5,
        "num_restarts": 6
    }

    fs = funm_krylov(A, b, param)

    exact = S @ jnp.diag(jnp.exp(EWs)) @ S.transpose() @ b
    print(exact)
    norms = jnp.linalg.norm(exact.reshape((n, 1)) - fs, axis=0)
    plt.plot(np.arange(param["num_restarts"]), norms)
    plt.show()
    print(norms)
    print(1 + 1)
