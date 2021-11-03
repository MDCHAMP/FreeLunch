'''
Base classes for optimisers

'''
from os import error
import numpy as np
from freelunch import darwin
from freelunch.adaptable import adaptable_set


class optimiser:
    '''
    Base class for all optimisers
    '''
    name = 'optimiser'
    tags = []
    hyper_definitions = {}
    hyper_defaults = {}
    can_run_quick = False

    def __init__(self, obj=None, hypers={}, bounds=None):
        '''
        Unlikely to instance the base class but idk what else goes here
        '''
        self.hypers = dict(self.hyper_defaults, **hypers) # Hyperparamters/ methods (dictionary of )
        self.bounds = np.array(bounds) # Bounds / constraints
        self.nfe = 0
        self.obj = self.wrap_obj_with_nfe(obj) # Objective function 

    def __call__(self, nruns=1, return_m=1, full_output=False):
        '''
        API for running the optimisation
        '''
        if self.obj is None:
            raise NotImplementedError('No optimiser selected')
        if nruns > 1:
            runs = [self.run() for i in range(nruns)]
            sols = np.concatenate(runs)
        else: 
            sols = self.run()
        sols = sorted(sols)
        if not full_output:
            return np.array([sol.dna for sol in sols[:return_m]])
        else:
            json_hypers = {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in self.hypers.items() }
            out = {
                'optimiser':self.name,
                'hypers':json_hypers,
                'bounds':self.bounds.tolist(),
                'nruns':nruns,
                'nfe':self.nfe,
                'solutions':[sol.dna.tolist() for sol in sols],
                'scores':[sol.fitness for sol in sols]
            }
            return out

    def __repr__(self):
        return self.name + ' optimisation object'    

    def run(self):
        if self.obj is None:
            raise NotImplementedError('No objective function selected')

    def run_quick(self):
        if self.obj is None:
            raise NotImplementedError('No objective function selected')

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

    def wrap_obj_with_nfe(self, obj):
        if obj is None: return None
        def w_obj(vec):
            self.nfe +=1
            fit = obj(vec)
            try:
                if np.isnan(fit): return None
                return float(fit)
            except(ValueError, TypeError):
                return None
        return w_obj

# Subclasses for granularity

class continuous_space_optimiser(optimiser):
    '''
    Base class for continuous space optimisers i.e DE, PSO, SA etc.
    '''

class discrete_space_optimiser(optimiser):
    '''
    Base class for discrete space optimisers i.e GA, GP etc.
    '''

