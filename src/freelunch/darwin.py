'''

Evolution is LIT 

    - Darwin, one can only presume

'''


import numpy as np

from freelunch.adaptable import adaptable_method
from freelunch.util import BadObjectiveFunctionScores


# %% Base class


class genetic_operation(adaptable_method):
    '''
    Base class for anything that happens while listening to Barry white...
    '''
    type = None
    name = 'genetic operation'
    hypers = {}

    def __init__(self, hypers=hypers):
        super().__init__()
        self.hypers = hypers

    def op(self, *parents, **hypers):
        raise NotImplementedError 


# %% Search operations
    

class rand_1(genetic_operation):
    type = 'mutation'
    name = 'rand/1'
    n_parents = 0
    hypers = {'F': 0.5}

    def op(self, *parents, pop=None, F=hypers['F']):
        idxs = np.random.randint(0, len(pop), 3)
        a, b, c = pop[idxs]
        # Assumes numpy array vectorisation
        return a.dna+ F * (b.dna - c.dna)


class rand_2(genetic_operation):
    type = 'mutation'
    name = 'rand/2'
    n_parents = 0
    hypers = {'F': 0.5}

    def op(self, *parents, pop=None, F=hypers['F']):
        idxs = np.random.randint(0, len(pop), 5)
        a, b, c, d, e = pop[idxs]
        return a.dna + F * (b.dna - c.dna) + F * (d.dna - e.dna)


class best_1(genetic_operation):
    type = 'mutation'
    name = 'best/1'
    n_parents = 0
    hypers = {'F': 0.5}

    def op(self, *parents, pop=None, F=hypers['F']):
        idxs = np.random.randint(0, len(pop), 2)
        a, b = pop[idxs]
        best = min(pop, key=lambda x: x.fitness)
        return best.dna + F * (a.dna - b.dna)


class best_2(genetic_operation):
    type = 'mutation'
    name = 'best/2'
    n_parents = 0
    hypers = {'F': 0.5}

    def op(self, *parents, pop=None, F=hypers['F']):
        idxs = np.random.randint(0, len(pop), 4)
        a, b, c, d = pop[idxs]
        best = min(pop, key=lambda x: x.fitness)
        return best.dna + F * (a.dna - b.dna) + F * (c.dna - d.dna)


class current_1(genetic_operation):
    type = 'mutation'
    name = 'current/1'
    n_parents = 1
    hypers = {'F': 0.5}

    def op(self, *parents, pop=None, F=hypers['F']):
        x = parents[0]
        idxs = np.random.randint(0, len(pop), 3)
        a, b, c = pop[idxs] 
        best = min(pop, key=lambda x: x.fitness)
        return x.dna + F * (best.dna - a.dna) + F * (b.dna - c.dna)



# %% Crossover operations


class binary_crossover(genetic_operation):
    type = 'crossover'
    name='binary crossover'
    hypers={'Cr':0.2}

    def op(self, parent1, parent2, Cr=hypers['Cr']):
        out = np.empty_like(parent1)
        for a, b, i in zip(parent1, parent2, range(len(parent1))):
            if np.random.uniform(0, 1) < Cr:
                out[i] = a
            else:
                out[i] = b
        jrand = np.random.randint(0, len(out)) #Ensure at least one difference
        out[jrand] = parent2[jrand]
        return out


# %% Selection operations


class binary_tournament(genetic_operation):
    '''
    2 - tournament selection
    '''
    type = 'selection'
    name = 'binary tournament'
    hypers={}

    def op(self, olds, news):
        out = np.empty_like(olds, dtype=object)
        for old, new, i in zip(olds, news, range(len(out))):
            if new < old:
                out[i] = new
                new.on_win()
            else:
                out[i] = old
        return out
