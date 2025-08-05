import numpy as np
from rank import tensor_rank_le, rationalise_factors


def build_matmul_tensor(n, m, p):
    A = np.zeros((n * m, m * p, n * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A[m * i + j, p * j + k, i * p + k] = 1
    return A


def check_factorisation(A, U, V, W, tol=1e-2):
    A_res = np.einsum('ia,ja,ka->ijk', U, V, W)
    assert np.linalg.norm(A_res - A) < tol


def test_matmul_2x2x2_7():
    A = build_matmul_tensor(2, 2, 2)
    success, (U, V, W) = tensor_rank_le(A, 7)
    assert success
    check_factorisation(A, U, V, W)


def test_matmul_3x3x3_23():
    A = build_matmul_tensor(3, 3, 3)
    success, (U, V, W) = tensor_rank_le(A, 23)
    assert success
    check_factorisation(A, U, V, W)


def test_matmul_2x2x2_6():
    A = build_matmul_tensor(2, 2, 2)
    success, _ = tensor_rank_le(A, 6)
    assert not success


def test_matmul_2x2x2_7_rationalise():
    A = build_matmul_tensor(2, 2, 2)
    success, (U, V, W) = tensor_rank_le(A, 7)
    assert success
    success, (U, V, W) = rationalise_factors(A, (U, V, W))
    assert success
    check_factorisation(A, U, V, W, tol=1e-9)
