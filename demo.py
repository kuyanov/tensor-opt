import numpy as np
from test_matmul import build_matmul_tensor
from rank import tensor_rank_le, rationalise_factors

if __name__ == '__main__':
    n, m, p, r = 3, 3, 3, 23
    A = build_matmul_tensor(n, m, p)

    success, (U, V, W) = tensor_rank_le(A, r, verbose=True)
    assert success
    A_res = np.einsum('ia,ja,ka->ijk', U, V, W)

    success, (U, V, W) = rationalise_factors(A, (U, V, W), verbose=True)
    assert success
    A_res = np.einsum('ia,ja,ka->ijk', U, V, W)
    print(f'error = {np.linalg.norm(A_res - A)}')
