import numpy as np
import scipy
from functools import partial
from semi_md.matrix_functions import *
from src.matfuncb.matfuncb import matfuncb

if __name__ == "__main__":
    A = np.random.random((3, 3))
    A = A + A.transpose()
    print(scipy.linalg.eigvals(A))
    n = A.shape[0]
    y = np.array([1.0, 2, 3, 0, 0, 0.01])


    # y'' = -A^2y
    # turn into x1' = x2; x2' = -A^2x1 with x1 = y and x2 = y'
    def deriv(t, y):
        return np.concatenate((y[n:], -(A @ A) @ y[:n]))


    t_end = 15
    result = scipy.integrate.solve_ivp(deriv, [0, t_end], y)
    print(f"Scipy integration result: {result.y[:n, -1]}, with derivative: {result.y[n:, -1]}")

    y2 = matfuncb(t_end ** 2 * A @ A, y[:n], cos_sqrtm, k=n, symmetric=True)[0] + \
         matfuncb(A, y[n:], partial(sinc_sqrtm_variation, t=t_end), k=n, symmetric=True)[0]
    d2 = -matfuncb(A @ A, y[:n], partial(xsinm, t=t_end), k=n, symmetric=True)[0] + \
         matfuncb(t_end ** 2 * A, y[n:], cos_sqrtm, k=n, symmetric=True)[0]
    print(f"My Matfunb result: {y2}, with derivative: {d2} and error: {np.linalg.norm(result.y[:n, -1] - y2)
                                                                       / np.linalg.norm(result.y[:n, -1])}")

    big_A = np.block([[np.zeros((n, n)), np.eye(n)], [-(A @ A), np.zeros((n, n))]])
    y3 = scipy.linalg.expm(t_end * big_A) @ y
    print(f"Expm result: {y3} with error: {np.linalg.norm(result.y[:n, -1] - y3[:n])
                                           / np.linalg.norm(result.y[:n, -1])}")
