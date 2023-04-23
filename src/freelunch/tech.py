"""Utility functions used by multiple optimisation algortihms.

All functions in this script should be unaware of optimiser state. 
"""

import numpy as np
from json import JSONEncoder

# %% Intialisation strategies


def uniform_continuous_init(bounds, N):
    return np.random.uniform(*bounds.T, (N, len(bounds)))


# %% Mutation strategies


def rand_1(pop, F, k=None):
    pos, _ = pop
    a, b, c = parent_idx_no_repeats(len(pos), n=3, k=k)
    return pos[a] + (F * (pos[b] - pos[c]))


def rand_2(pop, F, k=None):
    pos, _ = pop
    a, b, c, d, e = parent_idx_no_repeats(len(pos), n=5, k=k)
    return pos[a] + F * (pos[b] - pos[c]) + F * (pos[d] - pos[e])


def best_1(pop, F, k=None):
    pos, fit = pop
    a = np.argmin(fit)
    b, c = parent_idx_no_repeats(len(pos), n=2, k=k)
    return pos[a] + F * (pos[b] - pos[c])


def best_2(pop, F, k=None):
    pos, fit = pop
    a = np.argmin(fit)
    b, c, d, e = parent_idx_no_repeats(len(pos), n=4, k=k)
    return pos[a] + F * (pos[b] - pos[c]) + F * (pos[d] - pos[e])


def current_2(pop, F, k=None):
    pos, fit = pop
    a = np.argmin(fit)
    b, c, d = parent_idx_no_repeats(len(pos), n=3, k=k)
    if k is None:
        k = slice(None)
    return pos[k] + F * (pos[a] - pos[b]) + F * (pos[c] - pos[d])


# %% Crossover strategies


def binary_crossover(pos, newpos, Crs, jrand=True):
    idx = np.random.uniform(size=pos.shape) < Crs
    if jrand:  # always swap one dimension
        idx[:, np.random.randint(0, pos.shape[1])] = True * np.ones((pos.shape[0]))
    pos = pos.copy()
    pos[idx] = newpos[idx]
    return pos


# %% Selection strategies


def greedy_selection(oldpop, newpop, return_idx=False):
    idx = newpop[1] < oldpop[1]
    pop = oldpop[0].copy(), oldpop[1].copy()
    pop[0][idx] = newpop[0][idx]
    pop[1][idx] = newpop[1][idx]
    if return_idx:
        return (pop[0], pop[1]), idx
    else:
        return (pop[0], pop[1])


def update_best(best, newpop):
    idx = np.argmin(newpop[1])
    if newpop[1][idx] < best[1]:  # update best
        best = newpop[0][idx], newpop[1][idx]
    return best


# %% Bounding Strategies


def no_bounding(pos, bounds):
    raise Warning("No bounds applied")


def sticky_bounds(pos, bounds):
    out_low = pos < bounds[:, 0]
    out_high = pos > bounds[:, 1]
    pos = pos.copy()
    pos[out_low] = (np.ones_like(pos) * bounds[None, :, 0])[out_low]  # @TJR :chefs_kiss
    pos[out_high] = (np.ones_like(pos) * bounds[None, :, 1])[out_high]
    return pos


# %% Adaptable parameters


def lin_vary(lims, n, n_max):
    # Linearly vary with generations, e.g. inertia values
    return lims[1] + (lims[0] - lims[1]) * n / n_max


def normal_update(p, scores, jit=1e-6):
    if len(scores) == 0:
        out = p
    else:
        out = np.mean(scores), np.std(scores) + jit
    return out


def update_selection_probs(scores):
    hits, wins = scores
    p = (hits + 1) / (hits + wins)
    return p / np.sum(p)


def track_hits_wins(scores, selection_idx, success_idx):
    hits, wins = scores
    a, b = np.unique(selection_idx, return_counts=1)
    hits[a] += b
    for i in a:
        wins[i] += sum(success_idx[selection_idx == i])
    return hits, wins


# %%  Misc


def parent_idx_no_repeats(N, n, k=None):
    assert N > n
    if k is None:
        k = np.arange(N)
    elif type(k) is int:
        k = np.array([k])
    if len(k) < N / 10:  # for small samples it is more efficient to use choice
        return np.array(
            [np.random.choice(np.arange(N), size=(n), replace=False) for i in k]
        ).T
    else:  # for large samples we can go accross the population
        idxs = np.array([np.arange(N)] * (n + 1)).T
        for i in range(1, n + 1):
            while sum(same := np.any(idxs[:, :i] == idxs[:, None, i], axis=1)) > 0:
                idxs[same, i] = np.random.randint(0, N, sum(same))
    return idxs[k, 1:].T


def pdist(A, B=None):
    if B is None:
        B = A
    return np.sqrt(np.sum((A[:, None] - B[None, :]) ** 2, axis=-1))


def expm(A):
    v, S = np.linalg.eig(A)
    if not len(np.unique(v)) == len(v):
        if np.allclose(A, np.zeros_like(A)):  # zero case
            return np.eye(len(v))
        elif np.all(A[~np.diag(np.array([True] * len(v)))] == 0):  # matrix is diagonal
            return np.diag(np.exp(np.diag(A)))
        raise ValueError(
            "Non-diagonisable input matrix! Try choosing different parameters"
        )
    return np.real(S @ np.diag(np.exp(v)) @ np.linalg.inv(S))


class freelunch_json_encoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return JSONEncoder.default(self, obj)
