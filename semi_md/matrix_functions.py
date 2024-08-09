import scipy
import numpy as np


def sinc_sqrtm(A):
    """Calculate sinc(sqrt(A)). Note that if we want sinc(t sqrt(Omega^2)), the input has to be t^2 Omega^2."""
    w, v = scipy.linalg.eigh_tridiagonal(A.diagonal(), A.diagonal(-1))
    w = np.clip(w, 0, np.inf)
    # assert np.allclose(A @ v - v @ np.diag(w), np.zeros(A.shape))
    return v @ np.diag(np.sinc(np.sqrt(w) / np.pi)) @ v.T


def sinc2_sqrtm(A):
    """Calculate sinc^2(sqrt(A)). Note that if we want sinc^2(t sqrt(Omega^2)), the input has to be t^2 Omega^2."""
    w, v = scipy.linalg.eigh_tridiagonal(A.diagonal(), A.diagonal(-1))
    w = np.clip(w, 0, np.inf)
    # assert np.allclose(A @ v - v @ np.diag(w), np.zeros(A.shape))
    return v @ np.diag(np.sinc(np.sqrt(w) / np.pi) ** 2) @ v.T


def cos_sqrtm(A):
    """Calculate cos(sqrt(A)). Note that if we want cos(t sqrt(Omega^2)), the input has to be t^2 Omega^2."""
    w, v = scipy.linalg.eigh_tridiagonal(A.diagonal(), A.diagonal(-1))
    w = np.clip(w, 0, np.inf)
    # assert np.allclose(A @ v - v @ np.diag(w), np.zeros(A.shape))
    return v @ np.diag(np.cos(np.sqrt(w))) @ v.T


def sin_sqrtm(A):
    """Calculate sin(sqrt(A)). Note that if we want sin(t sqrt(Omega^2)), the input has to be t^2 Omega^2."""
    w, v = scipy.linalg.eigh_tridiagonal(A.diagonal(), A.diagonal(-1))
    w = np.clip(w, 0, np.inf)
    # assert np.allclose(A @ v - v @ np.diag(w), np.zeros(A.shape))
    return v @ np.diag(np.sin(np.sqrt(w))) @ v.T


def sqrtm(A):
    """Calculate sqrt(A). Note that if we want t sqrt(Omega^2), the input has to be t^2 Omega^2."""
    w, v = scipy.linalg.eigh_tridiagonal(A.diagonal(), A.diagonal(-1))
    w = np.clip(w, 0, np.inf)
    # assert np.allclose(A @ v - v @ np.diag(w), np.zeros(A.shape))
    return v @ np.diag(np.sqrt(w)) @ v.T


def sinc_sqrtm_variation(A, t):
    """Calculate A^(-1/2)sin(t sqrt(A)). Since this depends on t more explicitly the input is just A
    unlike for the other functions."""
    w, v = scipy.linalg.eigh_tridiagonal(A.diagonal(), A.diagonal(-1))
    w = np.clip(w, 0, np.inf)
    eps = np.finfo(t).eps
    vals = np.sin(t * np.sqrt(w) + eps) / (np.sqrt(w) + eps)  # Approximation to avoid the issues at 0
    # vals[w == 0] = 1
    # assert np.allclose(A @ v - v @ np.diag(w), np.zeros(A.shape))
    return v @ np.diag(vals) @ v.T


def sinc_sqrtm_variation3(A, t):
    """Calculate A^(-1/2)sin(t sqrt(A)). Since this depends on t more explicitly the input is just A
    unlike for the other functions.
    This version takes dense array A, like from non-symmetric matfuncb."""
    w, v = scipy.linalg.eigh(A)

    w = np.clip(w, 0, np.inf)
    eps = np.finfo(t).eps
    vals = np.sin(t * np.sqrt(w) + eps) / (np.sqrt(w) + eps)  # Approximation to avoid the issues at 0
    # vals[w == 0] = 1
    # assert np.allclose(A @ v - v @ np.diag(w), np.zeros(A.shape))
    return v @ np.diag(vals) @ v.T


def xsinm(A, t):
    """Calculate sqrt(A)sin(t sqrt(A)). Since this depends on t more explicitly the input is just A
    unlike for the other functions."""
    w, v = scipy.linalg.eigh_tridiagonal(A.diagonal(), A.diagonal(-1))
    w = np.clip(w, 0, np.inf)
    # assert np.allclose(A @ v - v @ np.diag(w), np.zeros(A.shape))
    return v @ np.diag(np.sqrt(w) * np.sin(t * np.sqrt(w))) @ v.T


def cosm_sincm(A):
    """Calculate cos(sqrt(A))sinc(sqrt(A)). Note that if we want
    cos(t sqrt(Omega^2))sinc(t sqrt(Omega^2)),
    the input has to be t^2 Omega^2."""
    w, v = scipy.linalg.eigh_tridiagonal(A.diagonal(), A.diagonal(-1))
    w = np.clip(w, 0, np.inf)
    # assert np.allclose(A @ v - v @ np.diag(w), np.zeros(A.shape))
    return v @ np.diag(np.cos(np.sqrt(w)) * np.sinc(np.sqrt(w) / np.pi)) @ v.T


def botchev_phi(A):
    """Calculate 2*(1-cos(sqrt(A))/A. """
    w, v = scipy.linalg.eig(A)
    # w = np.clip(w, 0, np.inf)
    return v @ np.diag(2 * (np.ones(len(w)) - np.cos(np.sqrt(w))) / w) @ v.T
