'''
Standard / common techniques that are used in several optimisers are abstracted to functions here. 
'''

# %% Imports

import numpy as np


# %% Custom exceptions

class BadObjectiveFunctionScores(Exception):
    '''Exception raised when both objective score comparisons evaluate false'''

class ZeroLengthSolutionError(Exception):
    '''Exception raised when an empty solution is passed to a benchmark'''



# %% Useful classes

class solution:
    '''
    Handy dandy common object for storing trial solutions / other interesting data
    '''
    def __init__(self, dna=None, fitness=None):
        self.dna = dna
        self.fitness = None 

class particle:
    '''
    Want to store info on particles in a swarm? I got you bud
    '''
    def __init__(self, pos=None, vel=None, fitness=None, best=None, best_pos=None):
        self.pos = pos
        self.vel = vel
        self._fitness = None
        self.fitness = fitness 
        self.best = best
        self.best_pos = best_pos

    @property
    def dna(self):
        return self.pos

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, fitness):
        if (fitness is not None) and ((self._fitness is None) or (fitness < self._fitness)):
            self.best = fitness
            self.best_pos = self.pos
        self._fitness = fitness

    def as_sol(self):
        sol = solution()
        sol.dna = self.best_pos
        sol.fitness = self.best
        return sol

class adaptable_parameter:
    '''
    Class for adaptable parameters for optimisers like SADE etc.
    '''
    pass



# %% Common methods

def uniform_continuous_init(bounds, N):
    out = np.empty((N,), dtype=object)
    for i in range(N):
        out[i] = solution(np.array([np.random.uniform(a,b) for a, b in bounds]))
    return out


def compute_obj(pop, obj):
    for sol in pop:
        sol.fitness = obj(sol.dna)
        if np.isnan(sol.fitness):
            sol.fitness = None

def binary_crossover(sol1, sol2, p):
    out = np.empty_like(sol1)
    for a,b,i in zip(sol1, sol2, range(len(sol1))):
        if np.random.uniform(0,1) < p:
            out[i] = a
        else:
            out[i] = b
    return out

def sotf(olds, news):
    out = np.empty_like(olds, dtype=object)
    for old, new, i in zip(olds, news, range(len(out))):
        if new.fitness < old.fitness:
            out[i] = new
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
        if dna[i] > high: out[i] = high
        elif dna[i] < low: out[i] = low
    return out




# %%
    