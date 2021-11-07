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


# def bounds_as_mat(bounds):
#     bounds_mat = np.zeros((len(bounds), 2))
#     for i, bound in enumerate(bounds):
#         bounds_mat[i, 0], bounds_mat[i, 1] = bound
#     return bounds_mat

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


#%% Bounding Strategies

class Bounder():

    hyper_defaults = {
        'eps': 1e-12, # Jitter
    }

    def __init__(self, bounds, hypers={}) -> None:
        self._bounds = None
        self.bounds = bounds
        self.hypers = dict(self.hyper_defaults, **hypers)

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, b):
        if isinstance(b, np.ndarray):
            self._bounds = b
        else:
            bounds_mat = np.zeros((len(b), 2))
            for i, bound in enumerate(b):
                bounds_mat[i, 0], bounds_mat[i, 1] = bound
            self._bounds = bounds_mat

    def bounding_function(self, pop):
        raise NotImplementedError

    def __call__(self, pop):
        for p in pop:
            self.bounding_function(p)

    def __iter__(self):
        return self.bounds.__iter__()

    def __len__(self):
        return self.bounds.shape[0]

    @property
    def shape(self):
        return self.bounds.shape

    def tolist(self):
        return {
            'strategy':self.__name__,
            'bounds': self.bounds.tolist(),
            'hypers': self.hypers
            }

class NoBounds(Bounder):
    '''
    Placeholder for no bounding
    '''

    def __init__(self) -> None:
        self.__name__ = 'No Bounds'
        pass

    def bounding_function(self, p):
        raise Warning('No bounds applied')

class StickyBounds(Bounder):
    '''
    Apply sticky bounds to space
    '''

    def __init__(self, bounds, hypers={}) -> None:
        self.__name__ = 'Sticky Bounds'
        super().__init__(bounds, hypers)

    def bounding_function(self, p):
        out = p.dna[:]
        for i, bound in enumerate(self.bounds):
            low, high = bound
            if p.dna[i] > high:
                out[i] = high - self.hypers['eps']
            elif p.dna[i] < low:
                out[i] = low + self.hypers['eps']
        p.dna = out
