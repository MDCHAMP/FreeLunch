'''

Evolution is LIT - Darwin, one can only presume

'''


import numpy as np


# %% Search operation classes and derivatives

class adaptable_search_operation():
    '''
    Base class for methods that are selected as part of an adaptive scheme
    Default behaviour is straight up reproduction yo
    '''
    name = 'Default adaptive search operation'
    n_parents = 1
    hypers = {}

    def __init__(self, hypers=hypers):
        self.hypers = hypers
        self.hits = [0]
        self.wins = [0]
        self.p = [1]

    def op(self, *parents, **hypers):
        return parents[0]

    def win(self):
        self.wins[-1] += 1

    def __call__(self, *parents, **hypers):
        self.hits[-1] += 1
        return self.op(*parents, **hypers)

    def update(self, p):
        self.hits.append(0)
        self.wins.append(0)
        self.p.append(p)

# Continuous sexual reproduction methods *Barry White starts playing*


class DE_rand_1(adaptable_search_operation):
    name = 'rand/1'
    n_parents = 0
    hypers = {'F': 0.5}

    def op(self, *parents, pop=None, F=hypers['F']):
        idxs = np.random.randint(0, len(pop), 3)
        a, b, c = pop[idxs]
        # Assumes numpy array vectorisation
        return a.dna+ F * (b.dna - c.dna)


class DE_rand_2(adaptable_search_operation):
    name = 'rand/2'
    n_parents = 0
    hypers = {'F': 0.5}

    def op(self, *parents, pop=None, F=hypers['F']):
        idxs = np.random.randint(0, len(pop), 5)
        a, b, c, d, e = pop[idxs]
        return a.dna + F * (b.dna - c.dna) + F * (d.dna - e.dna)


class DE_best_1(adaptable_search_operation):
    name = 'best/1'
    n_parents = 0
    hypers = {'F': 0.5}

    def op(self, *parents, pop=None, F=hypers['F']):
        idxs = np.random.randint(0, len(pop), 2)
        a, b = pop[idxs]
        best = min(pop, key=lambda x: x.fitness)
        # NOTE: This is a prime location to look for shitty bugs based on the use of min()
        return best.dna + F * (a.dna - b.dna)


class DE_best_2(adaptable_search_operation):
    name = 'best/2'
    n_parents = 0
    hypers = {'F': 0.5}

    def op(self, *parents, pop=None, F=hypers['F']):
        idxs = np.random.randint(0, len(pop), 4)
        a, b, c, d = pop[idxs]
        best = min(pop, key=lambda x: x.fitness)
        return best.dna + F * (a.dna - b.dna) + F * (c.dna - d.dna)


class DE_current_to_best_1(adaptable_search_operation):
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

class Crossover():
    ''' 
    Generic crossover operation
    '''

    name='Crossover'
    hypers={}

    def __init__(self,hypers=hypers):
        self.hypers = hypers

    def breed(self, parent1, parent2):
        raise NotImplementedError

class binary_crossover(Crossover):

    name='Binary Crossover'
    hypers={
        'Cr':0.2
    }

    def breed(self, parent1, parent2, p=hypers['Cr']):
        out = np.empty_like(parent1)
        for a, b, i in zip(parent1, parent2, range(len(parent1))):
            if np.random.uniform(0, 1) < p:
                out[i] = a
            else:
                out[i] = b
        #Ensure at least one difference
        jrand = np.random.randint(0, len(out))
        out[jrand] = parent2[jrand]
        return out