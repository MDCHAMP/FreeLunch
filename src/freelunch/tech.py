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


# %% Parameter classes and derivatives

class parameter:
    '''
    Boiler plate for parameter - 
    idk as far as best practice is concerned, 
    seems like nonsense to me.
    '''

    def __init__(self, value=None):
        self.value = value

    def __call__(self, *args):
        return self.op()

    def op(self):
        return self.value


class linearly_varying_parameter(parameter):

    def __init__(self, a0, an, n):
        self.a0 = a0
        self.an = an
        self.n = n
        self.values = np.linspace(a0, an, n)

    def op(self, k):
        return self.values[k]


class normally_varying_parameter(parameter):

    def __init__(self, u, sig):
        self.u = u
        self.sig = sig 
        self.value = np.random.normal(self.u, self.sig)

    def op(self):
        self.value = np.random.normal(self.u, self.sig)
        return self.value


class adaptable_normal_parameter(normally_varying_parameter):
    '''
    13th rule for life:
    meta-something > something
    '''

    def __init__(self, u, sig):
        super().__init__(u, sig)
        self.wins = []

    def win(self):
        self.wins.append(self.value)

    def update(self):
        if len(self.wins) > 0:
            self.u = np.mean(self.wins)
            self.sig = np.std(self.wins)
        self.wins = []


# %% Common methods

def uniform_continuous_init(bounds, N):
    out = np.empty((N,), dtype=object)
    for i in range(N):
        out[i] = animal(np.array([np.random.uniform(a, b)
                                    for a, b in bounds]))
    return out


def compute_obj(pop, obj):
    for sol in pop:
        sol.fitness = obj(sol.dna)
        if np.isnan(sol.fitness):
            sol.fitness = None
    return pop


def sotf(olds, news):
    out = np.empty_like(olds, dtype=object)
    for old, new, i in zip(olds, news, range(len(out))):
        print(old.dna, new.dna)
        if new.fitness < old.fitness:
            out[i] = new
            new.on_win()
        elif old.fitness <= new.fitness:
            out[i] = old
        else:
            raise BadObjectiveFunctionScores(
                'Winner could not be determined by comparing objective scores. scores:{} and {}'.format(
                    old.fitness, new.fitness
                ))
    return out

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


def pdist(A,B=None):
    '''
    Pairwise Euclidean Distance inside array
    '''
    
    if B is None:
        B = A

    return np.sqrt(np.sum((A[:,None]-B[None,:])**2,axis=-1))

