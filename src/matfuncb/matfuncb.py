import numpy as np
import scipy
from typing import Union

from src.matfuncb.np_funm import funm_krylov_v2, funm_krylov_v2_symmetric
from src.matfuncb.gershgorin import get_length_gershgorin
from src.matfuncb.power_method import get_length_power


def matfuncb(A: Union[np.array, scipy.sparse.sparray], b: np.array, f: Union[callable, str], k: int, symmetric: bool,
             accuracy:float, bound_method: Union[None, str]):
    """The central function to calculate the action of the matrix function f(A) on the vector b.

    :param A: The matrix.
    :param b: The vector.
    :param f: The matrix function, must accept matrix inputs.
    :param k: The size of the Krylov subspace.
    :param symmetric: Whether A is guaranteed to be symmetric, assumed to be False by default.

    For now let's make available these options:
    -f: exp as string; generic callable
    -A: np.array or sparse scipy; declared symmetric or not
    -k: explicit maximum Krylov size.
    Krylov size by a priori error bound

    Later:
    Krylov size by a priori error bound using a paused arnoldi decomposition
    Krylov size by a posteriori error bound
    Enabling restarts
    Evaluation by quadrature
    Evaluation by diagonalization

    Not part of this:
    """
    if isinstance(f, str):
        if f == "exp":
            f = scipy.linalg.expm
        else:
            raise NotImplemented(f"The function {f} is not implemented yet.")

    m = np.inf
    if bound_method == "gershgorin":
        m = get_length_gershgorin(A, accuracy)
    elif bound_method == "power":
        m = get_length_power(A, b, accuracy)
    eps = np.finfo(b.dtype).eps
    param = {"restart_length": min(k, m), "num_restarts": 1}
    if symmetric:
        fAb, _ = funm_krylov_v2_symmetric(A, b, f, k)
    else:
        fAb, _, _ = funm_krylov_v2(A, b, param, f, calculate_eigvals=False, stopping_acc=eps)
    return fAb.reshape(-1)
