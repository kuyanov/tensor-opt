import numpy as np
import torch
from descent import optimise, rationalise


def khatri_rao(U, V):
    return torch.einsum('ia,ja->ija', U, V).reshape((-1, U.shape[1]))


def last_factor(A3, U, V, eps=1e-3):
    rank = U.shape[1]
    UV = khatri_rao(U, V)
    return (torch.linalg.inv(UV.T @ UV + eps / 2 * torch.eye(rank)) @ UV.T @ A3).T


def get_all_factors(A: np.ndarray, factors: tuple[torch.tensor, torch.tensor], eps=1e-3):
    A3 = torch.reshape(torch.tensor(A), (-1, A.shape[2]))
    U, V = factors
    W = last_factor(A3, U, V, eps)
    return U.detach().numpy(), V.detach().numpy(), W.detach().numpy()


def tensor_rank_loss(A3, U, V, eps=1e-3):
    W = last_factor(A3, U, V, eps)
    UV = khatri_rao(U, V)
    return torch.linalg.norm(UV @ W.T - A3)


def tensor_rank_le(A: np.ndarray, r,
                   attempts=10,
                   num_iter=5000,
                   lr=0.5,
                   lr_decay=0.95,
                   eps=1e-3,
                   tol=1e-2,
                   verbose=False):
    A3 = torch.reshape(torch.tensor(A), (-1, A.shape[2]))
    factor_shapes = [(A.shape[0], r), (A.shape[1], r)]
    for _ in range(attempts):
        factors = optimise(factor_shapes,
                           lambda fac: tensor_rank_loss(A3, fac[0], fac[1]),
                           num_iter=num_iter, lr=lr, lr_decay=lr_decay, eps=eps, tol=tol,
                           dtype=A3.dtype, verbose=verbose)
        if factors is not None:
            return True, get_all_factors(A, factors)
    return False, (None, None, None)


def tensor_rank(A: np.ndarray, r_max=100):
    rank_l = 0
    rank_r = r_max
    factors = None
    while rank_r - rank_l > 1:
        rank_m = (rank_l + rank_r) // 2
        success, cur_factors = tensor_rank_le(A, rank_m)
        if success:
            rank_r = rank_m
            factors = cur_factors
        else:
            rank_l = rank_m
    return rank_r, factors


def rationalise_factors(A: np.ndarray, factors: tuple[np.ndarray, np.ndarray, ...],
                        num_iter=5000,
                        lr=0.1,
                        lr_decay=0.95,
                        eps=1e-3,
                        tol=1e-2,
                        denominators=(0, 1, 2, 3, 4, 5, 6, 8),
                        verbose=False):
    A3 = torch.reshape(torch.tensor(A), (-1, A.shape[2]))
    U = torch.tensor(factors[0], requires_grad=True)
    V = torch.tensor(factors[1], requires_grad=True)
    params = [U, V]
    success = rationalise(params,
                          lambda fac: tensor_rank_loss(A3, fac[0], fac[1]),
                          num_iter=num_iter, lr=lr, lr_decay=lr_decay, eps=eps, tol=tol,
                          denominators=denominators, verbose=verbose)
    return success, get_all_factors(A, tuple(params), eps=0)
