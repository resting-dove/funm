import numpy as np
import scipy
from src.matfuncb.matfuncb import matfuncb


def test_dense():
    A = np.random.random((5, 5))
    b = np.random.random(5)
    result, _ = matfuncb(A, b, "exp", 5)
    expected = scipy.linalg.expm(A).dot(b)
    print(result - expected)
    assert np.all(np.isclose(result, expected))


def test_sparse():
    A = scipy.sparse.random(5, 5, 0.1)
    b = np.random.random(5)
    result, _ = matfuncb(A, b, "exp", 5)
    expected = scipy.sparse.linalg.expm_multiply(A, b)
    print(result - expected)
    assert np.all(np.isclose(result, expected))


def test_callable():
    A = np.random.random((5, 5))
    b = np.random.random(5)
    result, _ = matfuncb(A, b, scipy.linalg.cosm, 5)
    expected = scipy.linalg.cosm(A).dot(b)
    print(result - expected)
    assert np.all(np.isclose(result, expected))


def test_symmetric():
    A = np.random.random((5, 5))
    A = A + A.T
    b = np.random.random(5)
    result, _ = matfuncb(A, b, scipy.sparse.linalg.expm, 5, symmetric=True)
    expected = scipy.linalg.expm(A).dot(b)
    print(result - expected)
    assert np.all(np.isclose(result, expected))

def test_lil_larger():
    n = 500
    A = np.random.random((n, n))
    A = A + A.T
    A = A / scipy.linalg.eigvalsh(A, subset_by_index=[499, 499])
    b = np.random.random(n)
    result, _ = matfuncb(A, b, scipy.sparse.linalg.expm, n, symmetric=True)
    expected = scipy.linalg.expm(A).dot(b)
    print(result - expected)
    assert np.all(np.isclose(result, expected))
    result, _ = matfuncb(A, b, scipy.sparse.linalg.expm, n, symmetric=False)
    print(result - expected)
    assert np.all(np.isclose(result, expected))

if __name__ == "__main__":
    test_lil_larger()
