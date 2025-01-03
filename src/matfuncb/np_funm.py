import numpy as np
from src.matfuncb.krylov_basis import extend_arnoldi, arnoldi
from src.matfuncb.error_bounds import get_length_gershgorin, get_length_power, expm_error_bound
from src.matfuncb.utils import get_eigvals_qr
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
        # V_big[:, k * m] = w

        (w, V_big, H, breakdown) = arnoldi(A=A, w=w, m=m)
        # (w, V_big, H, h, breakdown) = (
        #    jit(Arnoldi_2, static_argnames=["steps", "trunc", "reorth_num"])(A, V_big, H, s=k * m, steps=m, trunc=m))
        H_full[k * m: (k + 1) * m + 1, k * m: (k + 1) * m] = H

        H_exp = scipy.linalg.expm(H_full[: (k + 1) * m, : (k + 1) * m])
        H_exp_jax = np.array(H_exp)[-m:, 0]
        f = beta * (V_big[:, k * m: (k + 1) * m] @ H_exp_jax) + f
        fs[:, k] = f
        eigvals[k] = np.linalg.eigvals(H_full[:(k + 1) * m, :(k + 1) * m])
        update_norms.append(np.linalg.norm(beta * (V_big[:, k * m: (k + 1) * m] @ H_exp_jax)))
    return fs, eigvals, update_norms


def funm_krylov_v2(A, b: np.array, param, matfunc=scipy.linalg.expm, calculate_eigvals=True, stopping_acc=1e-10):
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
    current_size = 0
    for k in range(param["num_restarts"]):
        if stopping_criterion:
            break
        (w, V, H, breakdown) = arnoldi(A=A, w=w, m=m)
        if breakdown:
            print("breakdown")
            stopping_criterion = True
            m = breakdown
        H_full[current_size: current_size + m + 1, current_size: current_size + m] = H
        H_exp = matfunc(H_full[: current_size + m, : current_size + m])
        H_exp_jax = np.array(H_exp)[-m:, 0]
        f = beta * (V @ H_exp_jax) + f
        fs[:, k] = f
        update = np.linalg.norm(beta * V @ H_exp_jax)
        if calculate_eigvals:
            eigvals[k] = np.linalg.eigvals(H_full[:current_size + m, :current_size + m])
        update_norms.append(update)
        if update / np.linalg.norm(f) < stopping_acc:
            stopping_criterion = True
            print("Stopping accuracy reached.")
        if k > 10 and (update / update_norms[-1] < .05):
            stopping_criterion = True
            print("Updates getting to small.")
        current_size += m

    return fs, eigvals, update_norms, current_size + m


def lanczos_method(A, b: np.array, matfunc=scipy.sparse.linalg.expm, krylov_size: int = np.inf, *, max_starts: int = 1,
                   stopping_acc=1e-10, estimate_at: int = None, arnoldi_acc=1e-10, stopping_decay=0.05):
    """The symmetric variant of the function above.

    :param A: symmetric matrix.
    :param b: vector.
    :param matfunc: function that takes a matrix and returns a matrix. The function needs to be adapted to type of A.
    :param krylov_size: maximal size of the Krylov subspace.
    :param max_starts: Maximum number of starts.
    :param stopping_acc the desired accurarcy if a bound is used.
    :param estimate_at after how many steps to estimate the spectrum and then use the expm bound to derive a number of needed
        steps. This is semi a-priori. Later will enable a posteriori as well, which will change the signature again.
    :param arnoldi_acc: Parameter that's passed onto the Krylov basis generator on when to declare a breakdown.
    :param stopping_decay: Early stopping when updates get too small in relative norm.
        """
    assert krylov_size > 0
    assert max_starts >= 1
    if estimate_at and estimate_at > krylov_size:
        estimate_at = None
    stopping_criterion = False
    n = b.shape[0]
    beta = float(np.linalg.norm(b))
    w = b / beta
    m = krylov_size
    f = np.zeros((n, 1), dtype=b.dtype)
    fs = np.zeros((n, max_starts), dtype=b.dtype)
    HH = scipy.sparse.csc_array((0, 0),
                                dtype=b.dtype)  # ((krylov_size * max_starts + 2, krylov_size * max_starts), dtype=b.dtype)
    update_norms = []
    current_size = 0
    subdiag_array = None
    for k in range(max_starts):
        if stopping_criterion:
            fs = fs[:, :k]
            break
        (w, V, H, breakdown) = arnoldi(A=A, w=w, m=m, trunc=1, eps=arnoldi_acc)
        if breakdown:
            print("Breakdown")
            stopping_criterion = True
            m = breakdown
        HH = scipy.sparse.block_array(([HH, None], [subdiag_array, scipy.sparse.csc_array(H[:m, :m])]), format="csc")
        eta = H[m, m - 1]
        subdiag_array = fill_block_in_top_right(eta, rows=m, cols=HH.shape[1])
        H_exp = matfunc(HH)
        H_exp_jax = H_exp[-m:, [0]]
        f = beta * (V @ H_exp_jax) + f
        fs[:, k] = f[:, 0]
        update = np.linalg.norm(beta * V @ H_exp_jax)
        update_norms.append(update)
        if update / np.linalg.norm(f) < stopping_acc:
            stopping_criterion = True
            print("Stopping accuracy reached.")
        if k > 10 and (update / update_norms[-1] < stopping_decay):
            stopping_criterion = True
            print("Updates getting to small.")
        current_size += m

    return fs, update_norms, current_size


def fill_block_in_top_right(eta, rows, cols):
    return scipy.sparse.coo_array(([eta], ([0], [cols - 1])), shape=(rows, cols))


def gershgorin_adaptive_expm(A, b: np.array, calculate_eigvals=True, stopping_acc=1e-10):
    """Evaluation of exp(A)b using an adaptive krylov size.
    For now not restarted"""
    param = {"num_restarts": 1}
    m = get_length_gershgorin(A, stopping_acc)

    print(f"m is set to {m}.")
    param["restart_length"] = m
    fs, eigvals, update_norms, k = funm_krylov_v2(A, b, param, calculate_eigvals=calculate_eigvals,
                                                  stopping_acc=stopping_acc)
    return fs, eigvals, update_norms, k


def power_adaptive_expm(A, b: np.array, calculate_eigvals=True, stopping_acc=1e-10):
    """Evaluation of exp(A)b using an adaptive krylov size derived from a few ppower iteration steps.
    For now not restarted"""
    param = {"num_restarts": 1}
    m = get_length_power(A, b, stopping_acc)
    print(f"m is set to {m}.")
    param["restart_length"] = m
    fs, eigvals, update_norms, k = funm_krylov_v2(A, b, param, calculate_eigvals, stopping_acc)
    return fs, eigvals, update_norms, k
