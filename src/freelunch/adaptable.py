'''
Adaptable parameters and methods, functionality and API
'''

import numpy as np


# %% Adaptable methods


class adaptable_method():
    '''
    Base class for methods / parameters that are selected / updated as part of an adaptive scheme
    '''
    name = 'adaptable item'

    def __init__(self):
        self.hits = [0]
        self.wins = [0]
        self.p = [1]  # probability of selection

    def __call__(self, *parents, **hypers):
        self.hits[-1] += 1
        return self.op(*parents, **hypers)

    def win(self):
        self.wins[-1] += 1

    def reset_counts(self):
        self.hits.append(0)
        self.wins.append(0)

    def op(self,*parents, **hypers):
        raise NotImplementedError

class adaptable_set():
    '''
    Class for sets of competing methods i.e the mutation ops in SADE.

    Contains api for pinwheel selection and probability update
    '''
    name = 'adaptable set of methods'

    def __init__(self, strats=[]):
        self.strats = strats

    def select_strategy(self):
        '''
        API for pinwheel selection
        '''
        ps = [s.p[-1] for s in self.strats]
        ps = ps/np.sum(ps)
        return np.random.choice(self.strats, p=ps)

    def update_strategy_ps(self):
        '''
        API for probability update
        '''
        hits = np.array([s.hits[-1] for s in self.strats])
        wins = np.array([s.wins[-1] for s in self.strats])
        total_hits = np.sum(hits)
        total_wins = np.sum(wins)
        ps = np.zeros_like(hits)
        # update model - maybe abstract this at some point
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
        for strat, p in zip(self.strats, ps):
            strat.reset_counts()
            strat.p.append(p)


# %% Adaptable parameter classes and derivatives


class adaptable_parameter():
    '''
    Boiler plate for parameter
    '''

    def __init__(self, value=None):
        self.value = value
        self.hits = [0]
        self.wins = [0]
        self.win_values = []

    def __call__(self, *args):
        self.hits[-1] += 1
        return self.op(*args)

    def now(self):
        # return tuple of self and current value
        return self, self.value

    def op(self):
        return self.value

    def win(self, v):
        '''adaptable parameters store successful values'''
        self.win_values.append(v)
        self.wins[-1] += 1


class linearly_varying_parameter(adaptable_parameter):

    def __init__(self, a0, an, n):
        super().__init__()
        self.a0 = a0
        self.an = an
        self.n = n
        self.values = np.linspace(a0, an, n)

    def op(self, k):
        return self.values[k]


class normally_varying_parameter(adaptable_parameter):

    def __init__(self, u, sig):
        super().__init__()
        self.u = u
        self.sig = sig
        self.value = np.random.normal(self.u, self.sig)

    def op(self):
        self.value = np.random.normal(self.u, self.sig)
        return self.value

    def update(self):  # fit normal distribution to successful parameters
        u = self.u if len(self.win_values) == 0 else np.mean(self.win_values)
        std = self.std if len(self.win_values) == 0 else max(np.std(self.win_values), 10**-3)
        if not np.isnan(u):
            self.u = u 
        else:
            raise ValueError # Stop don't just stay still
        if not np.isnan(std):
            self.std = std
        else:
            raise ValueError # Stop don't just stay still
        self.win_values = [] # Not storing all winning parameters at the moment
        self.wins.append(0)
        self.hits.append(0)
