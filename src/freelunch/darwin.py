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


# Exportable dictionary
DE_methods = {x.name: x for x in [
    DE_rand_1, DE_rand_2, DE_best_1, DE_best_2, DE_current_to_best_1]}


# API for probability update
def update_strategy_ps(strats):
    hits = np.array([s.hits[-1] for s in strats])
    wins = np.array([s.wins[-1] for s in strats])
    total_hits = np.sum(hits)
    total_wins = np.sum(wins)
    ps = np.zeros_like(hits)
    # update model
    # prevent zero division with a bit of fun bias!!
    dem = np.sum(wins * (hits + wins)) + 1
    for i, h, w in zip(range(len(ps)), hits, wins):
        num = h * (total_hits + total_wins)
        ps[i] = num / dem
    # normalise
    n = np.sum(ps)
    if n == 0:
        ps += 1
    ps = ps / n
    for strat, p in zip(strats, ps):
        strat.update(p)


# API for strategy selection 
def select_strategy(strats):
    ps = [s.p[-1] for s in strats]
    ps = ps/np.sum(ps)
    return np.random.choice(strats, p=ps)