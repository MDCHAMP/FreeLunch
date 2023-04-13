"""
Base classes for optimisers

"""
import json
from functools import partial
from typing import Iterable

import numpy as np

from freelunch import tech

_BAD_OBJ_SCORE = 1e308


class optimiser:
    """
    Base class for all optimisers
    """

    name = "optimiser"
    tags = []
    hyper_definitions = {"N": "Population size", "G": "Number of generations"}
    hyper_defaults = {"N": 0, "G": 1}

    def __init__(self, obj, bounds=None, hypers={}):
        """
        Unlikely to instance the base class but idk what else goes here
        """
        # Bounding
        self.bounds = bounds  # Bounds / constraints
        if bounds is None:
            self.bounder = tech.no_bounding
        elif isinstance(bounds, Iterable):
            self.bounder = tech.sticky_bounds
        # Objective funciton
        self.nfe = 0
        self.obj = partial(self.wrap_obj, obj)
        # Hyperparamters/ methods
        self.hypers = self.hyper_defaults | hypers

    def __call__(self, n_runs=1):
        """
        API for running the optimisation
        """
        best = (None, 0)
        runs = []
        for n in range(n_runs):
            self.run()
            runs.append(self._to_dict())
            b = runs[n]["pos"][0], runs[n]["fit"][0]
            if n == 0 or b[1] < best[1]:
                best = b
        return best, runs

    def wrap_obj(self, obj, vec):
        """Adds nfe counting and bad value handling to raw_obj"""
        fit = obj(vec)
        self.nfe += 1
        if not isinstance(fit, float):
            fit = _BAD_OBJ_SCORE
        return fit

    def pre_loop(self):
        """Here is where each optimiser sets up before looping"""
        ...

    def post_loop(self):
        """Here is where each optimiser cleans up after looping"""
        ...

    def step(self):
        """Placeholder optimisation step"""
        ...

    def post_step(self):
        ...

    def run(self):
        """Generic Run Loop
        We want to implement a common interface for all optimisers.
        """
        self.nfe, self.gen = 0, 0
        self.pre_loop()
        self.post_step()
        # Main Loop
        for self.gen in range(1, self.hypers["G"]):
            # Step the optimiser
            self.step()
            if self.post_step() is False:
                break
    
        self.post_loop()

    def _to_dict(self):
        idx = np.argsort(self.fit)
        return {
            "bounds": self.bounds,
            "hypers": self.hypers,
            "pos": self.pos[idx],
            "fit": self.fit[idx],
            "nfe": self.nfe,
        }
