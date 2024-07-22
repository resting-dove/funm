import numpy as np
import scipy
from typing import Union

from src.matfuncb.np_funm import funm_krylov_v2, funm_krylov_v2_symmetric
from src.matfuncb.error_bounds import get_length_gershgorin, get_length_power


def matfuncb(A: Union[np.array, scipy.sparse.sparray], b: np.array, f: Union[callable, str], k: int, symmetric= False,
             accuracy:float = 1e-20, bound_method: str = ""):
    """The central function to calculate the action of the matrix function f(A) on the vector b.

    :param A: The matrix.
    :param b: The vector.
    :param f: The matrix function, must accept matrix inputs.
    :param k: The size of the Krylov subspace.
    :param symmetric: Whether A is guaranteed to be symmetric.
    :param accuracy: The desired accuracy, to be used in combination with a bound. By default a value smaller than
            double precision eps.
    :param bound_method: The name of the method to derive a bound on the necessary Krylov size.
            Current options are "gershgorin" or "power" concatenated with "pos" or "neg", or "semi-a-priori".
            The first part chooses the bound method, both of which assume that the matrix has real eigenvalues in one
            of the halves of the reals. The second part declares which one. The default is positive.

    For now let's make available these options:
    -f: exp as string; generic callable
    -A: np.array or sparse scipy; declared symmetric or not
    -k: explicit maximum Krylov size.
    Krylov size by a priori error bound
    Krylov size by a priori error bound using a paused arnoldi decomposition

    Later:
    Krylov size by a priori error bound for more adaptive spectra
    Krylov size by a posteriori error bound
    Enabling restarts
    Evaluation by quadrature
    Evaluation by diagonalization

    Krylov size by a priori error bound for non symmetric matrix

    Not part of this:
    """
    if isinstance(f, str):
        if f == "exp":
            f = scipy.linalg.expm
        else:
            raise NotImplemented(f"The function {f} is not implemented yet.")
    info = {}
    bound = None
    m = np.inf
    if "gershgorin" in bound_method:
        # gershgorin bound assumes the spectrum is in [0, b] and estimates b
        sign = int(not "neg" in bound_method) * 2 - 1
        m = get_length_gershgorin(sign * A, accuracy)
    elif "power" in bound_method:
        # TODO: This could be pushed down into the method?
        # power bound assumes the spectrum is in [0, b] and estimates b
        sign = int(not "neg" in bound_method) * 2 - 1
        m = get_length_power(sign * A, b, accuracy)
    elif "semi-a-priori" in bound_method:
        if not symmetric:
            raise NotImplementedError("Semi a-priori krylo bound not implemented for non symmetric matrices yet.")
        bound = 2
    info["bound_krylov_size"] = m
    eps = np.finfo(b.dtype).eps
    k = min(k, m)
    info["max_krylov_size"] = k
    param = {"restart_length": k, "num_restarts": 1}
    if symmetric:
        fAb, _, k = funm_krylov_v2_symmetric(A, b, f, k, stopping_acc=accuracy, bound=bound)
    else:
        fAb, _, _, k = funm_krylov_v2(A, b, param, f, calculate_eigvals=False, stopping_acc=eps)
    info["actual_krylov_size"] = k
    return fAb.reshape(-1), info
