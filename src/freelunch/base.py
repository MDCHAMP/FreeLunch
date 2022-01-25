'''
Base classes for optimisers

''' 
from functools import partial
from multiprocessing import Pool

import numpy as np

from freelunch import darwin
from freelunch import tech
from freelunch.adaptable import adaptable_set
from freelunch.util import UnpicklableObjectiveFunction

class optimiser:
    '''
    Base class for all optimisers
    '''
    name = 'optimiser'
    tags = []
    hyper_definitions = {}
    hyper_defaults = {}
    can_run_quick = False

    def __init__(self, obj, hypers={}, bounds=None):
        '''
        Unlikely to instance the base class but idk what else goes here
        '''        
        self.bounds = bounds # Bounds / constraints
        self.bounder = tech.sticky_bounds
        self.nfe = 0
        self.obj = partial(self._obj, obj)  # Extend capability of objective function (mp safe!!)
        self.hypers = dict(self.hyper_defaults, **hypers) # Hyperparamters/ methods 
        self.hypers['bounding'] = self.bounder.__name__

    def __call__(self, n_runs=1, n_return=1, full_output=False, n_workers=1, pool_args={}, chunks=1):
        '''
        API for running the optimisation
        '''
        if n_runs > 1:
            if n_workers > 1:
                try:
                    ret = Pool(n_workers, **pool_args).starmap(self.run_mp, [() for _ in range(n_runs)], chunks)
                except AttributeError:
                    raise UnpicklableObjectiveFunction
                runs, _nfes = [list(a) for a in zip(*ret)]
                self.nfe = sum(_nfes)
            else:
                runs = [self.run() for i in range(n_runs)]
            sols = np.concatenate(runs)
        else: 
            sols = self.run()
        sols = sorted(sols)
        if not full_output:
            return np.array([sol.dna for sol in sols[:n_return]])
        else:
            json_hypers = {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in self.hypers.items() }
            out = {
                'optimiser':self.name,
                'hypers':json_hypers,
                'bounds':self.bounds.tolist(),
                'nruns':n_runs,
                'nfe':self.nfe,
                'solutions':[sol.dna.tolist() for sol in sols],
                'scores':[sol.fitness for sol in sols]
            }
            return out

    def __repr__(self):
        return self.name + ' optimisation object'

    def run_mp(self):
        return self.run(), self.nfe

    def parse_hyper(self, op):
        if isinstance(op, list): # top 10 recursive gamer moments
            strats = [self.parse_hyper(strat) for strat in op]
            return adaptable_set(strats)
        elif isinstance(op,type) and issubclass(op, darwin.genetic_operation):
            return op()
        elif isinstance(op, str):
            try:
                op = getattr(darwin,op)
                return op()
            except AttributeError:
                raise AttributeError('Method not recognised, refer to docs for list of implemented methods') # TODO test

    def apply_bounds(self, pop, **hypers):
        '''Apply the bouding method to every object in an iterable'''
        for sol in pop:
            self.bounder(sol, self.bounds, **hypers)
    

    def _obj(self, obj, vec):
        '''Adds nfe counting and bad value handling to raw_obj'''
        self.nfe +=1
        fit = obj(vec)
        try:
            if np.isnan(fit): return None
            return float(fit)
        except(ValueError, TypeError):
            return None


# Subclasses for granularity

class continuous_space_optimiser(optimiser):
    '''
    Base class for continuous space optimisers i.e DE, PSO, SA etc.
    '''

class discrete_space_optimiser(optimiser):
    '''
    Base class for discrete space optimisers i.e GA, GP etc.
    '''

