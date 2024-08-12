import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

from src.matfuncb.error_bounds import get_length_gershgorin, get_length_power
from src.matfuncb.matfuncb import matfuncb
from test_np.utils.matrix_factory import get_symmetric_matrix_by_evals

# Investigate how the error bounds behave for a matrix


if __name__ == "__main__":
    evals = -1 * np.array([2000, 20, 20, 20, 20, 0, 0, 0] + list(np.arange(0, 1, 0.01)))
    Omega2, S = get_symmetric_matrix_by_evals(evals, True)
    b = np.random.random(Omega2.shape[0])

    exact = S.T @ np.diag(np.exp(evals)) @ S @ b
    max_acc = 1e-16
    errors = []
    i_s = []
    for i in range(1, int(len(evals) / 2)):
        app, info = matfuncb(Omega2, b, scipy.sparse.linalg.expm, k=i, symmetric=True)
        err = np.linalg.norm(app - exact)
        errors.append(err)
        i_s.append(i)
        if err <= max_acc:
            break
    plt.plot(i_s, errors)
    for acc in [1e-5, 1e-10, 1e-15]:
        m = get_length_gershgorin(-Omega2, acc)
        ger = plt.scatter(m, acc, label="Ger", marker="o", c="b")
        m = get_length_power(-Omega2, b, acc)
        pow = plt.scatter(m, acc, label="Pow", marker="x", c="r")

        app, info = matfuncb(Omega2, b, scipy.sparse.linalg.expm, k=len(evals), symmetric=True, accuracy=acc,
                             bound_method="semi-a-priori")
        err = np.linalg.norm(app - exact)
        semi = plt.scatter(info["actual_krylov_size"], err, label="Semi", marker="D", c="g")
    plt.yscale("log")
    plt.legend(handles=[ger, pow, semi])
    plt.show()
