"""Utility functions used by multiple optimisation algortihms.

All functions in this script should be unaware of optimiser state. 
"""
import warnings
from json import JSONEncoder

import numpy as np

# %% Intialisation strategies


def uniform_continuous_init(bounds, N):
    """Generate a size N population uniformly within bounds.

    Args:
        bounds (np.array): (D, 2) array of [lower, upper] bounds where D is the problem dimension.
        N (int): Size of population to be generated

    Returns:
        np.ndarray: (N, D) sampled positions.
    """
    return np.random.uniform(*bounds.T, (N, len(bounds)))


# %% Mutation strategies


def rand_1(pop, F, k=None):
    """DE/rand/1 mutation.

    For details see: https://ieeexplore.ieee.org/abstract/document/4632146 (eqn 2)

    Args:
        pop (tuple): (pos, fit) arrays of parent population.
        F (float):  Mutation parameter
        k (int, optional): Population indicies, len(k) new vectors are generated. A value of None indicates N new vectors should be generated. Defaults to None.

    Returns:
        np.ndarray: New vectors computed from previous population.
    """
    pos, _ = pop
    a, b, c = parent_idx_no_repeats(len(pos), n=3, k=k)
    return pos[a] + (F * (pos[b] - pos[c]))


def rand_2(pop, F, k=None):
    """DE/rand/2 mutation.

    For details see: https://ieeexplore.ieee.org/abstract/document/4632146 (eqn 6)

    Args:
        pop (tuple): (pos, fit) arrays of parent population.
        F (float):  Mutation parameter
        k (int, optional): Population indicies, len(k) new vectors are generated. A value of None indicates N new vectors should be generated. Defaults to None.

    Returns:
        np.ndarray: New vectors computed from previous population.
    """
    pos, _ = pop
    a, b, c, d, e = parent_idx_no_repeats(len(pos), n=5, k=k)
    return pos[a] + F * (pos[b] - pos[c]) + F * (pos[d] - pos[e])


def best_1(pop, F, k=None):
    """DE/best/1 mutation.

    For details see: https://ieeexplore.ieee.org/abstract/document/4632146 (eqn 3)

    Args:
        pop (tuple): (pos, fit) arrays of parent population.
        F (float):  Mutation parameter
        k (int, optional): Population indicies, len(k) new vectors are generated. A value of None indicates N new vectors should be generated. Defaults to None.

    Returns:
        np.ndarray: New vectors computed from previous population.
    """
    pos, fit = pop
    a = np.argmin(fit)
    b, c = parent_idx_no_repeats(len(pos), n=2, k=k)
    return pos[a] + F * (pos[b] - pos[c])


def best_2(pop, F, k=None):
    """DE/best/2 mutation.

    For details see: https://ieeexplore.ieee.org/abstract/document/4632146 (eqn 5)

    Args:
        pop (tuple): (pos, fit) arrays of parent population.
        F (float):  Mutation parameter
        k (int, optional): Population indicies, len(k) new vectors are generated. A value of None indicates N new vectors should be generated. Defaults to None.

    Returns:
        np.ndarray: New vectors computed from previous population.
    """
    pos, fit = pop
    a = np.argmin(fit)
    b, c, d, e = parent_idx_no_repeats(len(pos), n=4, k=k)
    return pos[a] + F * (pos[b] - pos[c]) + F * (pos[d] - pos[e])


def rand_to_best_1(pop, F, k=None):
    """DE/rand-to-best/1 mutation.

    For details see: https://ieeexplore.ieee.org/abstract/document/4632146 (eqn 4)

    Args:
        pop (tuple): (pos, fit) arrays of parent population.
        F (float):  Mutation parameter
        k (int, optional): Population indicies, len(k) new vectors are generated. A value of None indicates N new vectors should be generated. Defaults to None.

    Returns:
        np.ndarray: New vectors computed from previous population.
    """
    pos, fit = pop
    a = np.argmin(fit)
    b, c, d = parent_idx_no_repeats(len(pos), n=3, k=k)
    if k is None:
        k = slice(None)
    return pos[k] + F * (pos[a] - pos[b]) + F * (pos[c] - pos[d])


# %% Crossover strategies


def binary_crossover(pos, newpos, Crs, jrand=True):
    """Binary crossover with probability Cr.

    Args:
        pos (np.ndarray): (N, D) array of current population.
        newpos (_type_): _description_
        Crs (_type_): _description_
        jrand (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    idx = np.random.uniform(size=pos.shape) < Crs
    if jrand:  # always swap one dimension
        idx[:, np.random.randint(0, pos.shape[1])] = True * np.ones((pos.shape[0]))
    pos = pos.copy()
    pos[idx] = newpos[idx]
    return pos


# %% Selection strategies


def greedy_selection(oldpop, newpop, return_idx=False):
    """Binary tournament selction.

    Args:
        oldpop (tuple): (pos, fit) arrays of previous population.
        newpop (tuple): (pos, fit) arrays of new population.
        return_idx (bool, optional): Wether or not the idxs of new entrants should be returned. Defaults to False.

    Returns:
       tuple: (pos, fit) arrays of next population.
    """
    idx = newpop[1] < oldpop[1]
    pop = oldpop[0].copy(), oldpop[1].copy()
    pop[0][idx] = newpop[0][idx]
    pop[1][idx] = newpop[1][idx]
    if return_idx:
        return (pop[0], pop[1]), idx
    else:
        return (pop[0], pop[1])


def update_best(best, newpop):
    """Track best position and fitness

    Args:
        best (tuple): (pos, fit) of current best.
        newpop (tuple): (pos, fit) arrays of new population.

    Returns:
        tuple: (pos, fit) of new best.
    """
    idx = np.argmin(newpop[1])
    if newpop[1][idx] < best[1]:  # update best
        best = newpop[0][idx], newpop[1][idx]
    return best


# %% Bounding Strategies


def no_bounding(pos, bounds):
    """Do not apply any bounding

    Args:
        pos (np.ndarray): (N, D) array of current population.
        bounds (np.array): (D, 2) array of [lower, upper] bounds where D is the problem dimension.

    Raises:
        Warning: Will warn the user that there is no bounding.

    Returns:
        np.ndarray: (N, D) array of (un)bounded population.
    """
    warnings.warn("No bounds applied")
    return pos.copy()


def sticky_bounds(pos, bounds):
    """Clip vectors to the edges of the search space.

    Args:
        pos (np.ndarray): (N, D) array of current population.
        bounds (np.array): (D, 2) array of [lower, upper] bounds where D is the problem dimension.

    Returns:
        np.ndarray: (N, D) array of bounded population.
    """
    return np.clip(pos.copy(), *bounds.T)


# %% Adaptable parameters


def lin_vary(lims, t, T):
    """Get the value of a linearly varing parameter.

    Args:
        lims (Iterable): y(0), y(T).
        t (int): Current timestep.
        T (int): Maximum timesteps.

    Returns:
        float: Value of the parameter at the current timestep.
    """
    return lims[1] + (lims[0] - lims[1]) * t / T


def normal_update(p, scores, jit=1e-6):
    """Update the parameters of a 1D Gassian distribution from samples.

    Args:
        p (tuple): (mu, sig), current values of the parameters
        scores (Iterable): Samples of the random variable.
        jit (_type_, optional): Numerical jitter added to std. Defaults to 1e-6.

    Returns:
        tuple: New values of the parameters.
    """
    if len(scores) == 0:
        out = p
    else:
        out = np.mean(scores), np.std(scores) + jit
    return out


def update_selection_probs(hits, wins):
    """Update the probability of selecting seach strategy.

    For more details see:  https://ieeexplore.ieee.org/abstract/document/4632146 (eqn 14)

    Args:
        hits (np.ndarray): Number of times each strategy was selected in the previous learning period.
        wins (np.ndarray): Number of times each strategy generated a successful vector in the previous learning period.

    Returns:
        np.array: Updated mutation proabailities.
    """
    p = (hits + 1) / (hits + wins)
    return p / np.sum(p)


def track_hits_wins(scores, selection_idx, success_idx):
    """Update hits and wins from selection and success idx.

    Args:
        scores (tuple): (hits, wins) running totals of the hits and wins.
        selection_idx (np.ndarray): idx of the selected methods.
        success_idx (_type_): idx of the successful methods.

    Returns:
        tuple: (hits, wins) running totals of the hits and wins.
    """
    hits, wins = scores
    a, b = np.unique(selection_idx, return_counts=1)
    hits[a] += b
    for i in a:
        wins[i] += sum(success_idx[selection_idx == i])
    return hits, wins


# %%  Misc


def parent_idx_no_repeats(N, n, k=None):
    """Generate n sets of n integers in [0, N] such that no two elements in a row are the same.

    This implenetation heuristically decides wehter to compute the sets row-wise or columnwise based on the ratio of k and N.

    Args:
        N (int): Population size.
        n (int): Number of parent index rows to generate.
        k (int, optional): Size of parent vectors. If None, k=N is used. Defaults to None.

    Returns:
        np.ndarray: (n, k) array of parent indicies
    """    
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
    """Pairwise distances between two sets of vectors

    Args:
        A (np.ndarray): First set of vectors.
        B (np.ndarray, optional): Second set of vectors. If None, defaults to A. Defaults to None.

    Returns:
        np.ndarray: pairwise differneces.
    """
    if B is None:
        B = A
    return np.sqrt(np.sum((A[:, None] - B[None, :]) ** 2, axis=-1))


def expm(A):
    """Matric exponential.

    https://en.wikipedia.org/wiki/Matrix_exponential

    Args:
        A (np.ndarray): Input matrix.

    Raises:
        ValueError: Matrix exponential cannot be computed.

    Returns:
        np.ndarray: Matrix exponential of the input.
    """
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
    """Custom encoder for freelunch that parses np.ndarrays."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return JSONEncoder.default(self, obj)
