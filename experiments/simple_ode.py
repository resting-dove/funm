import numpy as np
import scipy
from src.matfuncb.matfuncb import matfuncb

def deriv_vec(t, y):
    return A @ y

if __name__ == "__main__":

    A = np.array([[-0.25,   0,    0.33],
                  [0.25,    -0.2, 0],
                  [0,       0.2,  -0.1]])
    print(scipy.linalg.eigvals(A))
    y = np.array([10.0, 20, 30])
    # y' = Ay, which is solved by y = exp(tA)y_0
    t_end = 15
    result = scipy.integrate.solve_ivp(deriv_vec, [0, t_end], y)
    print(f"Scipy integration result: {result.y[:, -1]}")
    y2 = matfuncb(t_end * A, y, 'exp', k=3)
    print(f"My Matfunb result: {y2[0]}")
    print(f"Expm result: {scipy.linalg.expm(t_end * A) @ y}")
