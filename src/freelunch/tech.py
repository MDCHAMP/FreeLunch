'''
Standard / common techniques that are used in several optimisers are abstracted to functions here. 
'''

# %% Imports

import numpy as np
from freelunch.zoo import animal

# %% Custom exceptions


class BadObjectiveFunctionScores(Exception):
    '''Exception raised when both objective score comparisons evaluate false'''


class ZeroLengthSolutionError(Exception):
    '''Exception raised when an empty solution is passed to a benchmark'''


class SolutionCollapseError(Exception):
    '''Exception raised when all solutions are identical'''


# %% Common methods

def uniform_continuous_init(bounds, N, creature=animal):
    out = np.empty((N,), dtype=object)
    for i in range(N):
        adam = creature()
        adam.dna = np.array([np.random.uniform(a, b)
                             for a, b in bounds])
        out[i] = adam
    return out


def compute_obj(pop, obj):
    for sol in pop:
        sol.fitness = obj(sol.dna)
        if np.isnan(sol.fitness):
            sol.fitness = None
    return pop


def apply_sticky_bounds(dna, bounds):
    out = dna[:]
    for i, bound in enumerate(bounds):
        low, high = bound
        if dna[i] > high:
            out[i] = high
        elif dna[i] < low:
            out[i] = low
    return out


def bounds_as_mat(bounds):
    bounds_mat = np.zeros((len(bounds), 2))
    for i, bound in enumerate(bounds):
        bounds_mat[i, 0], bounds_mat[i, 1] = bound
    return bounds_mat


def lin_reduce(lims, n, n_max):
    # Linearly reduce with generations, e.g. inertia values
    return lims[1] + (lims[0]-lims[1])*n/n_max


def pdist(A, B=None):
    '''
    Pairwise Euclidean Distance inside array
    '''
    if B is None:
        B = A
    return np.sqrt(np.sum((A[:, None]-B[None, :])**2, axis=-1))
