'''
Standard / common techniques that are used in several optimisers are abstracted to functions here. 
'''

# %% Imports

import numpy as np



# %% Common methods

def update_local_best(opt):
    better = opt.fit < opt.local_best_fit
    opt.local_best_fit[better] = opt.fit[better] 
    opt.local_best_pos[better] = opt.pos[better]   

def update_global_best(opt):
    gbest = np.argmin(opt.local_best)
    opt.global_best_fit = opt.fit[gbest]
    opt.global_best_pos = opt.local_best_pos[gbest]


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


#%% Bounding Strategies


def no_bounding(p, bounds, **hypers):
    '''
    Placeholder for no bounding
    '''
    raise Warning('No bounds applied')

def sticky_bounds(p, bounds, eps=1e-12, **hypers): # TODO vectorise
    '''
    Apply sticky bounds to space
    '''
    out = p.dna[:]
    for i, bound in enumerate(bounds):
        low, high = bound
        if   p.dna[i] > high: out[i] = high - eps
        elif p.dna[i] < low:  out[i] = low + eps
        p.dna = out

# %% Intialisation strategies

def uniform_continuous_init(bounds, N):
    return np.random.uniform(*bounds.T,  (N, len(bounds)))

def gaussian_neigbourhood_init(bounds, N, creature, mu=None, sig=None): # TODO sensibleise
    if mu is None:
        mu = [(a+b)/2 for a,b in bounds]
    if sig is None:
        sig = [(b-a)/6 for a,b in bounds]
    out = np.empty((N,), dtype=object)
    for i in range(N):
        adam = creature()
        adam.dna = np.random.normal(mu, sig)
        out[i] = adam
    return out