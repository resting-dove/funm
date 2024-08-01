import numpy as np
import scipy
from src.matfuncb.matfuncb import matfuncb

def deriv_vec(t, y):
    return A @ y + g(y)

def phi(A):
    A_inv = scipy.linalg.inv(A)
    return (scipy.linalg.expm(A) - np.eye(A.shape[0])) @ A_inv

if __name__ == "__main__":
    n = 40
    t_end = 0.1
    A = np.random.random((n, n))
    print(scipy.linalg.expm_cond(t_end * A))
    y_0 = np.random.random(n)
    def g(y):
        y_0 = y + np.random.normal(0, 1, size=y.shape)
        r_0 = 0.1
        k = 0.1
        return k * (np.linalg.norm(y - y_0) - r_0) * (y_0 - y)
    # y' = Ay + g(y), which is solved exactly by
    # y = exp(tA)y_0 + \int_0^t exp((t-\tau)A) g(y) d\tau
    # Using an approximation to the integral this can be calculated as
    # y = \exp(tA)y_0 + t \phi_1(tA)g(y_0),
    # with \phi_1(tA) = (\exp(tA) - 1)/(tA)
    result = scipy.integrate.solve_ivp(deriv_vec, [0, t_end], y_0)
    print(f"Scipy integration result: {result.y[:, -1]}")
    y2 = matfuncb(t_end * A, y_0, 'exp', k=n)[0]
    y2 = y2 + t_end * matfuncb(t_end * A, g(y_0), phi, k=3)[0]
    print(f"My Matfunb result: {y2}, error: {np.linalg.norm(result.y[:, -1] - y2)
                                             / np.linalg.norm(result.y[:, -1])}")
    y3 = scipy.linalg.expm(t_end * A) @ y_0
    A_inv = scipy.linalg.inv(A)
    y3 = (y3 + t_end * (scipy.linalg.expm(t_end * A) - np.eye(A.shape[0]))
          @ (A_inv / t_end) @ g(y_0))
    print(f"Expm result: {y3}, error: {np.linalg.norm(result.y[:, -1] - y3) 
                                            / np.linalg.norm(result.y[:, -1])}")
