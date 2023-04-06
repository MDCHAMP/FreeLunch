'''
Standard / common techniques that are used in several optimisers are abstracted to functions here. 
'''

# %% Imports

import numpy as np

# %% Common methods

def greedy_selection(old_fit, new_fit, old_vars, new_vars):
    idx = new_fit < old_fit
    old_fit[idx] = new_fit[idx]
    if isinstance(old_vars, np.ndarray):
        old_vars[idx] = new_vars[idx]
    else:
        for o, n in zip(old_vars,new_vars):
            o[idx] = n[idx]


def update_local_best(opt):
    better = opt.fit < opt.local_best_fit
    opt.local_best_fit[better] = opt.fit[better] 
    opt.local_best_pos[better] = opt.pos[better]   

def update_global_best(opt):
    gbest = np.argmin(opt.local_best_fit)
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


def no_bounding(opt):
    '''
    Placeholder for no bounding
    '''
    raise Warning('No bounds applied')

def sticky_bounds(opt): # TODO vectorise
    '''
    Apply sticky bounds to space
    '''

    out_low = opt.pos < opt.bounds[:,0]
    out_high = opt.pos > opt.bounds[:,1]
    opt.pos[out_low] = np.tile(opt.bounds[:,0],[opt.pos.shape[0],1])[out_low]
    opt.pos[out_high] = np.tile(opt.bounds[:,1],[opt.pos.shape[0],1])[out_high]


    # out = p.dna[:]
    # for i, bound in enumerate(bounds):
    #     low, high = bound
    #     if   p.dna[i] > high: out[i] = high - eps
    #     elif p.dna[i] < low:  out[i] = low + eps
    #     p.dna = out

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