import numpy as np
import scipy

from src.matfuncb.krylov_basis import arnoldi


class MatfuncEvaluator():
    """Class to enable the efficient evaluation of f(A)b and g(A)b."""

    def __init__(self):
        self.v = None
        self.w = None

    def diagonalize(self, A, override=False):
        if self.w is None or override:
            self.w, self.v = scipy.linalg.eigh(A.todense())
            self.w = np.clip(self.w, a_min=0, a_max=np.infty)

    def diagonalize_tri(self, A, override=False):
        if override:
            self.w, self.v = scipy.linalg.eigh_tridiagonal(A.diagonal(), A.diagonal(-1))
            self.w = np.clip(self.w, a_min=0, a_max=np.infty)

    def lanczos_diagonalize(self, A, b, override=False):
        if override:
            self.beta = scipy.linalg.norm(b)
            (v, V, H, breakdown) = arnoldi(A, b.flatten() / self.beta, 100, trunc=1)
            self.V = V
            if breakdown is False:
                m = 100
            else:
                m = breakdown
            self.H = H[:m, :m]
            self.diagonalize_tri(self.H, override=True)

    def sym_cosm_sqrt_l(self, h: float, omega2: scipy.sparse.sparray, b: np.array) -> np.array:
        self.lanczos_diagonalize(omega2, b, override=True)
        fw = np.diag(np.cos(h * np.sqrt(self.w)))
        fH = (self.v @ (fw @ (self.v.T)))[:, [0]]
        f = self.beta * (self.V @ fH)
        return f

    def sym_sincm_sqrt_l(self, h: float, omega2: scipy.sparse.sparray, b: np.array) -> np.array:
        self.lanczos_diagonalize(omega2, b)
        fw = np.diag(np.sinc(h * np.sqrt(self.w) / np.pi))
        fH = (self.v @ (fw @ (self.v.T)))[:, [0]]
        f = self.beta * (self.V @ fH)
        return f

    def sym_msinm_sqrt_l(self, h: float, omega2: scipy.sparse.sparray, b: np.array) -> np.array:
        self.lanczos_diagonalize(omega2, b)
        fw = np.diag(np.sqrt(self.w) * np.sin(h * np.sqrt(self.w)))
        fH = (self.v @ (fw @ (self.v.T)))[:, [0]]
        f = self.beta * (self.V @ fH)
        return f

    def sym_cosm_sqrt(self, h: float, omega2: scipy.sparse.sparray, b: np.array) -> np.array:
        # w, v = scipy.linalg.eigh(omega2.todense())
        self.diagonalize(omega2, override=True)
        fw = np.diag(np.cos(h * np.sqrt(self.w)))
        return self.v @ (fw @ (self.v.T @ b))

    def sym_sincm_sqrt(self, h: float, omega2: scipy.sparse.sparray, b: np.array) -> np.array:
        # w, v = scipy.linalg.eigh(omega2.todense())
        self.diagonalize(omega2)
        fw = np.diag(np.sinc(h * np.sqrt(self.w) / np.pi))
        return self.v @ (fw @ (self.v.T @ b))

    def sym_msinm_sqrt(self, h: float, omega2: scipy.sparse.sparray, b: np.array) -> np.array:
        # w, v = scipy.linalg.eigh(omega2.todense())
        self.diagonalize(omega2)
        fw = np.diag(np.sqrt(self.w) * np.sin(h * np.sqrt(self.w)))
        return self.v @ (fw @ (self.v.T @ b))
