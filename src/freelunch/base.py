'''
Base classes for optimisers

'''
from functools import partial
from typing import Iterable

import numpy as np
from freelunch import tech


class optimiser:
    '''
    Base class for all optimisers
    '''
    name = 'optimiser'
    tags = []
    hyper_definitions = {}
    hyper_defaults = {}
    can_run_quick = False

    def __init__(self, obj, bounds=None, hypers={}):
        '''
        Unlikely to instance the base class but idk what else goes here
        '''

        # Bounding
        self.bounds  = bounds  # Bounds / constraints
        if bounds is None:
            self.bounder = tech.no_bounding 
        elif isinstance(bounds, Iterable):
            self.bounder = tech.sticky_bounds
        
        # Objective funciton
        self.nfe = 0
        self.obj = partial(self.wrap_obj, obj)
        
        # Hyperparamters/ methods
        self.hypers = self.hyper_defaults | hypers

        # Hooks
        self.post_step = None

    def __call__(self):
        '''
        API for running the optimisation
        '''

        self.run()
        idx = np.argmin(self.fit)
        return self.fit[idx], self.pos[idx]


    def wrap_obj(self, obj, vec):
        '''Adds nfe counting and bad value handling to raw_obj'''
        self.nfe += 1
        fit = obj(vec)
        # fit = verify_well_behaved(fit)
        return fit

    def pre_loop(self):
        """Here is where each optimiser sets up before looping
        """
        ...
        
    def step(self):
        """Placeholder optimisation step
        """
        # This placeholder is causing test_no_optimiser to fail.
        # @TODO: Raise warning instead?
        ... 
        
    def post_step_hook(self):
        if self.post_step is not None:
            return self.post_step(self)

    def run(self):
        """Generic Run Loop
        We want to implement a common interface for all optimisers.
        """
        # All methods initalise a population 
        self.gen = 0
        self.pre_loop()
        self.post_step_hook()
        # Main Loop
        for self.gen in range(1,self.hypers["G"]):
            # Step the optimiser
            self.step()
            # Provide a point to hook in after the step
            if self.post_step_hook() is False:
                break

