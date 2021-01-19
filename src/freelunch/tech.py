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

class SolutionCollapseError(Exception):
    '''Exception raised when all solutions are identical'''




# %% Useful classes

class solution:
    '''
    Handy dandy common object for storing trial solutions / other interesting data
    '''
    def __init__(self, dna=None, fitness=None):
        self.dna = dna
        self.fitness = None 

class particle(solution):
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
        if (fitness is not None) and ((self._fitness is None) or (fitness < self.best)):
            self.best = fitness
            self.best_pos = self.pos
        self._fitness = fitness

    def as_sol(self):
        sol = solution()
        sol.dna = self.best_pos
        sol.fitness = self.best
        return sol

class krill(particle):

    '''
    I am a krill, a type of animal (maybe implemented later), and a type of solution

    I am also basically just a particle...
    '''

    def __init__(self, pos=None, vel=None, motion=None, forage=None, fitness=None, best=None, best_pos=None):
        super().__init__(pos, vel, fitness, best, best_pos)
        self.motion = motion
        self.forage = forage

    




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
    return pop

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

def bounds_as_mat(bounds):
    bounds_mat = np.zeros((len(bounds),2))
    for i,bound in enumerate(bounds):
        bounds_mat[i,0], bounds_mat[i,1] = bound
    return bounds_mat

def lin_reduce(lims,n,n_max):
    # Linearly reduce with generations, e.g. inertia values
    return lims[1] + (lims[0]-lims[1])*n/n_max



# %%
    