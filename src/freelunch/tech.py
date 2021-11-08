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


def uniform_continuous_init_shy(bounds, N, creature=animal, r=1):
    points = []
    i = 0
    while len(points) < N:
        i += 1
        if i > 200: # give up and try a different configuration
            out = []
        trial = np.array([np.random.uniform(a, b) for a, b in bounds])
        if len(points) == 0 or all([np.linalg.norm(trial-a) > r for a in points]):
            points.append(trial)
    out = np.empty((N,), dtype=object)
    for i, point in enumerate(points):
        adam = creature()
        adam.dna = point
        out[i] = adam
    return out


def compute_obj(pop, obj):
    for sol in pop:
        sol.fitness = obj(sol.dna)
    return pop


def bounds_as_mat(bounds):
    bounds_mat = np.zeros((len(bounds), 2))
    for i, bound in enumerate(bounds):
        bounds_mat[i, 0], bounds_mat[i, 1] = bound
    return bounds_mat



def lin_reduce(lims, n, n_max):
    # Linearly reduce with generations, e.g. inertia values
    if lims[1] < lims[0]: 
        if isinstance(lims,list):
            lims.reverse()
        else:
            np.flip(lims)
    return lims[1] + (lims[0]-lims[1])*n/n_max


def scale_obj(obj, bounds, u=0, d=2):
    # affine scaling of obj function
    us = np.array([(high + low)/2 for low, high in bounds])
    ds = np.array([(high - low)/2 for low, high in bounds])
    def obj_scaled(x):
        x_scaled = (x - us) / ds
        return obj(x_scaled)
    def unscaler(x_scaled):
        return (x_scaled * ds) + us
    return obj_scaled, unscaler


def pdist(A, B=None):
    '''
    Pairwise Euclidean Distance inside array
    '''
    if B is None:
        B = A
    return np.sqrt(np.sum((A[:, None]-B[None, :])**2, axis=-1))


#%% Bounding Strategies


def no_bounding(p, bounds, **hypers):
    '''
    Placeholder for no bounding
    '''
    raise Warning('No bounds applied')

def sticky_bounds(p, bounds, eps=1e-12, **hypers):
    '''
    Apply sticky bounds to space
    '''
    out = p.dna[:]
    for i, bound in enumerate(bounds):
        low, high = bound
        if   p.dna[i] > high: out[i] = high - eps
        elif p.dna[i] < low:  out[i] = low + eps
        p.dna = out
