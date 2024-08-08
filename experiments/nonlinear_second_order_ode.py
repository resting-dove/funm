import numpy as np
import scipy
from functools import partial
from semi_md.matrix_functions import *
from src.matfuncb.matfuncb import matfuncb

if __name__ == "__main__":

    A = np.random.random((3, 3))
    A = A + A.transpose()
    A = A @ A
    print(scipy.linalg.eigvals(A))
    n = A.shape[0]
    y = np.array([1.0, 2, 3, 0.1, 0.1, 0.01])


    def g(y):
        y_0 = y[:n] + np.random.normal(0, 1, size=n)
        r_0 = 0.1
        k = 0.1
        return k * (np.linalg.norm(y - y_0) - r_0) * (y_0 - y)


    # y'' = -A^2y + g(y)
    # turn into x1' = x2; x2' = -A^2 x1 + g(x1) with x1 = y and x2 = y'
    def deriv(t, y):
        return np.concatenate((y[n:], -A @ y[:n] + g(y[:n])))


    t_end = 15
    result = scipy.integrate.solve_ivp(deriv, [0, t_end], y)
    print(f"Scipy integration result: {result.y[:n, -1]}, with derivative: {result.y[n:, -1]}")

    y2 = y
    steps = 30
    t_step = t_end / steps
    for i in range(1, 1 + steps):
        x1 = matfuncb(t_step ** 2 * A, y2[:n], cos_sqrtm, k=n, symmetric=True)[0] + \
             matfuncb(A, y2[n:], partial(sinc_sqrtm_variation, t=t_step), k=n, symmetric=True)[0]
        Lambda = g(matfuncb(t_step ** 2 * A, y2[:n], sinc_sqrtm, k=n, symmetric=True)[0])
        x1 = x1 + 0.5 * t_step ** 2 * matfuncb(t_step ** 2 * A, Lambda, sinc2_sqrtm, k=n, symmetric=True)[0]
        x2 = -matfuncb(A, y2[:n], partial(xsinm, t=t_step), k=n, symmetric=True)[0] + \
             matfuncb(t_step ** 2 * A, y2[n:], cos_sqrtm, k=n, symmetric=True)[0]
        Lambda2 = g(matfuncb(t_step ** 2 * A, x1, sinc_sqrtm, k=n, symmetric=True)[0])
        term = matfuncb(t_step ** 2 * A, Lambda, cosm_sincm, k=n, symmetric=True)[0] + \
               matfuncb(t_step ** 2 * A, Lambda2, sinc_sqrtm, k=n, symmetric=True)[0]
        x2 = x2 + 0.5 * t_step * term
        y2 = np.concatenate((x1, x2))
    print(f"My Matfunb result with {steps} steps: {y2[:n]} with error: {np.linalg.norm(result.y[:n, -1] - y2[:n])
                                                                        / np.linalg.norm(result.y[:n, -1])}")
    y2 = y
    steps = 2
    t_step = t_end / steps
    for i in range(1, 1 + steps):
        x1 = matfuncb(t_step ** 2 * A, y2[:n], cos_sqrtm, k=n, symmetric=True)[0] + \
             matfuncb(A, y2[n:], partial(sinc_sqrtm_variation, t=t_step), k=n, symmetric=True)[0]
        Lambda = g(matfuncb(t_step ** 2 * A, y2[:n], sinc_sqrtm, k=n, symmetric=True)[0])
        x1 = x1 + 0.5 * t_step ** 2 * matfuncb(t_step ** 2 * A, Lambda, sinc2_sqrtm, k=n, symmetric=True)[0]
        x2 = -matfuncb(A, y2[:n], partial(xsinm, t=t_step), k=n, symmetric=True)[0] + \
             matfuncb(t_step ** 2 * A, y2[n:], cos_sqrtm, k=n, symmetric=True)[0]
        Lambda2 = g(matfuncb(t_step ** 2 * A, x1, sinc_sqrtm, k=n, symmetric=True)[0])
        term = matfuncb(t_step ** 2 * A, Lambda, cosm_sincm, k=n, symmetric=True)[0] + \
               matfuncb(t_step ** 2 * A, Lambda2, sinc_sqrtm, k=n, symmetric=True)[0]
        x2 = x2 + 0.5 * t_step * term
        y2 = np.concatenate((x1, x2))
    print(f"My Matfunb result with {steps} steps: {y2[:n]} with error: {np.linalg.norm(result.y[:n, -1] - y2[:n])
                                                                        / np.linalg.norm(result.y[:n, -1])}")
