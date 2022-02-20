'''
Utility functions
'''

import numpy as np

# %% Custom exceptions

class BadObjectiveFunctionScores(Exception):
    '''Exception raised when both objective score comparisons evaluate false'''

class UnpicklableObjectiveFunction(Exception):
    '''Exception raised when objectivefunction cannot be pickled for multiprocessing'''
class ZeroLengthSolutionError(Exception):
    '''Exception raised when an empty solution is passed to a benchmark'''
class InvalidSolutionUpdate(Exception):
    '''Exception raised when trying to move an animal/particle to a bad location'''

# %% Custom warnings

class SolutionCollapseWarning(Warning):
    '''Warning: All solutions in the population are identical'''

class KrillSingularityWarning(SolutionCollapseWarning):
    '''Warning: Krill singularity identified, adding pertubation'''

#%% Helper functions

def real_finite(a):

    if a is None:
        return
    elif not isinstance(a, float) and not isinstance(a,int):
        raise ValueError
    elif not np.isreal(a):
        raise ValueError
    elif np.isnan(a):
        raise ValueError
    elif a == np.inf or a == -np.inf:
        raise ValueError


#%% Decorators

def verify_real_finite(check_args, check_vars):

    def wrapped_func(func):
        def wrapper(*args, **kwargs):
            for a in check_args:
                real_finite(args[a])
            for v in check_vars:
                try:
                    real_finite(kwargs[v])
                except KeyError:
                    pass
            func(*args, **kwargs)
        return wrapper
    return wrapped_func