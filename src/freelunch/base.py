"""Base classes for optimisers.

Description of module goes here...

"""
from functools import partial
from typing import Iterable
from multiprocessing import Pool
import numpy as np
from freelunch import tech

_BAD_OBJ_SCORE = 1e308


class optimiser:
    """Base class for all optimisation methods.

    Implement basic functionality common to all optimisation classes including wrapping the objective function, `pre_run` and `post_step` methods and the call API. This class also sets the hyperparameters and sets the bounding method heuristically by parsing the bounds argument.

    Attributes:
        name: Name of optimisation algortihm (from the paper).
        tags: Keywords for the optimisation algorithm
        hyper_definitions: Definitions of any hyperparameters in the algortihm
        hyper_defaults: Default values of hyperparameters from the source paper (unless otherwise stated)
        obj: Objective function to be optimised. Note that freelunch always assumes a minimisation problem.
        bounds: A Dx2 array of [lower, upper] bounds where D is the problem dimension.
        hypers: Hyperparameters of the optimisation algorthim (to override the defaults)
        nfe: Number of function evaluations in current run
        pos: Positions of current population (NxD)
        fit: Fitness scores of current population (N,)
        global best: Tuple of (pos, fit) the position and fitness of the best evaluation
    """

    name = "optimiser"
    tags = []
    hyper_definitions = {"N": "Population size", "G": "Number of generations"}
    hyper_defaults = {"N": 100, "G": 100}

    def __init__(self, obj, bounds=None, hypers={}):
        """Instance the optimiser.

        Instance the optimiser and set the bounding method and hyperparameters. This method also wraps the objective function so that bad values and nfe counting is handled automatically.

        Args:
            obj (callable): The objective function to be optimised
            bounds ([np.ndarray, None], optional): A Dx2 array of [lower, upper] bounds where D is the problem dimension. Defaults to None.
            hypers (dict, optional): Dictionary of hyperparameters to be overwritten. Defaults to {}.
        """
        # Bounding
        self.bounds = bounds  # Bounds / constraints
        if bounds is None:
            self.bounder = tech.no_bounding
        elif isinstance(bounds, Iterable):
            self.bounder = tech.sticky_bounds
        # Objective funciton
        self.nfe = 0
        self.obj = partial(self._wrap_obj, obj)
        self.global_best = (None, _BAD_OBJ_SCORE)
        # Hyperparamters/ methods
        self.hypers = optimiser.hyper_defaults | self.hyper_defaults | hypers
        self.post_step_hook = None

    def __call__(self, n_runs=1, n_workers=1, mp_args={}):
        """Run the optimisation.

        Args:
            n_runs (int, optional): Number of times to run the optimisation. Defaults to 1.
            n_workers (int, optional): Number of processes to run the optimisation on. Defaults to 1 (no parallelisation).
            mp_args ():
        Returns:
            Tuple: Tuple of (pos, fit) for the best solution in n_runs
            List: List of dict with data from each of the `n_runs`. See `optimiser._to_dict` for details.
        """
        # MP case
        if n_workers > 1:
            with Pool(processes=n_workers, **mp_args) as pool:
                runs = pool.starmap(self.run, [()]*n_runs)
        # No MP
        else:
            runs = [self.run() for _ in range(n_runs)]
        # post process
        for run in runs:
            if run["best"][1] < self.global_best[1]:
                self.global_best = run["best"]
        return self.global_best, runs

    def run(self):
        """Generic framework for an optimisation run.

        In FreeLunch, the run of each algorithm is standardised and several common methods are called. All optimisers proceed in the following manner:

        Before the loop the following mehtods are called:
        - `optimiser.pre_loop`
        - `update global best`
        - `optimiser.post_step`

        Each iteration the following are called:
        - `optimiser.step`
        - `track global best`
        - `optimiser.post_step` (Loop will break if this returns False)

        After the loop the following mehtods are called:
        - `optimiser.post_loop`

        Most custom behaviour can be achieved by overwriting one or more of these methods.
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
        # Post loop
        self.post_loop()
        return self._to_dict()

    def pre_loop(self):
        """Logic to be executed before the main loop."""
        pass

    def step(self):
        """Logic to be executed during each step of the optimiser.

        This method updates the optimiser.pos and optimiser.fit attrs.
        """
        pass

    def post_step(self):
        """Logic to be executed after each iteration of the main loop.

        Returns:
            Any: Return flag. If False, optimisation halts
        """        
        if self.post_step_hook is not None:
            return self.post_step_hook(self)

    def post_loop(self):
        """Logic to be executed after the main loop optimiser (i.e cleanup, postprocessing)."""
        pass

    def _wrap_obj(self, obj, vec):
        """Wrap the objective function for nfe counting and bad score handling.

        Args:
            obj (callable): The objective funciton to be minimised.
            vec (np.ndarray): The parameter vector to be evaluated.

        Returns:
            callable: The wrapped objective function.
        """
        fit = obj(vec)
        self.nfe += 1
        # validation checks go left to right an only evaluate if the prev passes
        if (
            fit is None
            or isinstance(fit, bool)
            or not np.isfinite(fit)
            or not np.isreal(fit)
        ):
            fit = _BAD_OBJ_SCORE
        if fit < self.global_best[1]:
            self.global_best = vec, fit
        return fit

    def _to_dict(self):
        """Deposit run information into a dictionary.

        Returns:
            dict: Summary of final state after optimisation.
        """
        idx = np.argsort(self.fit)
        return {
            "best": self.global_best,
            "bounds": self.bounds,
            "hypers": self.hypers,
            "pos": self.pos[idx].copy(),
            "fit": self.fit[idx].copy(),
            "nfe": self.nfe,
        }
