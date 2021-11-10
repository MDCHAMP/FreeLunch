'''
Standard / common techniques that are used in several optimisers are abstracted to functions here. 
'''

# %% Imports

import numpy as np
from freelunch.zoo import animal


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
    return pop


def apply_sticky_bounds(dna, bounds):
    eps = 1e-12
    out = dna[:]
    for i, bound in enumerate(bounds):
        low, high = bound
        if dna[i] > high:
            out[i] = high - eps
        elif dna[i] < low:
            out[i] = low + eps
    return out


def bounds_as_mat(bounds):
    bounds_mat = np.zeros((len(bounds), 2))
    for i, bound in enumerate(bounds):
        bounds_mat[i, 0], bounds_mat[i, 1] = bound
    return bounds_mat

    # MDCHAMP am I missing something here TR
    # @TR I think naievely calling the constructor on nested iterables is deprecieated iirc and might 
    # not behave like we would expect 
    # i.e np.array([np.array([0, 1, ..]), np.array([0, 1, ...])]) =/= np.array([[0,1,...], [0,1,...]]) 
    # The above would fail a structural equality test for example and report different dtypes. 
    # return np.array(bounds)


def lin_reduce(lims, n, n_max):
    # Linearly reduce with generations, e.g. inertia values
    if lims[1] < lims[0]: 
        if isinstance(lims,list):
            lims.reverse()
        else:
            np.flip(lims)
    return lims[1] + (lims[0]-lims[1])*n/n_max


def pdist(A, B=None):
    '''
    Pairwise Euclidean Distance inside array
    '''
    if B is None:
        B = A
    return np.sqrt(np.sum((A[:, None]-B[None, :])**2, axis=-1))
