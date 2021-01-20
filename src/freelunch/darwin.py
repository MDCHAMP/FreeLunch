'''

Evolution is LIT - Darwin, one can only presume

'''


import numpy as np


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
        self.value

    def op(self):
        self.value = np.random.normal(self.u, self.sig)
        return self.value


class adaptable_normal_parameter(normally_varying_parameter):
    '''
    13th rule for life:
    meta-something > something
    '''

    def __init__(self, u, sig):
        super().__init__(self, u, sig)
        self.wins = []

    def win(self):
        self.hits.append(self.value)

    def update(self):
        self.u = np.mean(self.hits)
        self.sig = np.std(self.hits)
        self.wins = []


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

    def op(self, *parents, pop=None):
        return parents[0]

    def win(self):
        self.wins[:-1] += 1

    def __call__(self, *parents, pop=None):
        self.hits[-1] += 1
        return self.op(*parents, pop=None)

    def update(self, p):
        self.hits.append(0)
        self.wins.append(0)
        self.p.append(p)

# Continuous sexual reproduction methods *Barry White starts playing*


class DE_rand_1(adaptable_search_operation):
    name = 'rand/1/bin'
    n_parents = 0
    hypers = {'F': 0.5}

    def op(self, *parents, pop=None, F=hypers['F']):
        a, b, c = np.random.choice(pop, 3)
        # Assumes numpy array vectorisation
        return a + F * (b - c)


class DE_rand_2(adaptable_search_operation):
    name = 'rand/2/bin'
    n_parents = 0
    hypers = {'F': 0.5}

    def op(self, *parents, pop=None, F=hypers['F']):
        a, b, c, d, e = np.random.choice(pop, 5)
        return a + F * (b - c) + F * (d - e)


class DE_best_1(adaptable_search_operation):
    name = 'best/1/bin'
    n_parents = 0
    hypers = {'F': 0.5}

    def op(self, *parents, pop=None, F=hypers['F']):
        a, b = np.random.choice(pop, 2)
        best = min(pop, key=lambda x: x.fitness)
        # NOTE: This is a prime location to look for shitty bugs based on the use of min()
        return best + F * (a - b)


class DE_best_2(adaptable_search_operation):
    name = 'best/2/bin'
    n_parents = 0
    hypers = {'F': 0.5}

    def op(self, *parents, pop=None, F=hypers['F']):
        a, b, c, d = np.random.choice(pop, 4)
        best = min(pop, key=lambda x: x.fitness)
        return best + F * (a - b) + F * (c - d)


class DE_current_to_best_1(adaptable_search_operation):
    name = 'current/1/bin'
    n_parents = 2
    hypers = {'F': 0.5}

    def op(self, *parents, pop=None, F=hypers['F']):
        x = parents[:1]
        a, b, c = np.random.choice(pop, 3)
        best = min(pop, key=lambda x: x.fitness)
        return x + F * (best - a) + F * (b - c)


# Exportable dictionary
DE_methods = {x.name: x for x in [
    DE_rand_1, DE_rand_2, DE_best_1, DE_best_2, DE_current_to_best_1]}

# API for probability update


def update_strategy_p(strats):
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
