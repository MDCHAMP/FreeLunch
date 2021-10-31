'''
Utility functions
'''

import numpy as np
from numpy.lib.type_check import real

# %% Custom exceptions
class BadObjectiveFunctionScores(Exception):
    '''Exception raised when both objective score comparisons evaluate false'''

class ZeroLengthSolutionError(Exception):
    '''Exception raised when an empty solution is passed to a benchmark'''

class SolutionCollapseError(Exception):
    '''Exception raised when all solutions are identical'''

class InvalidSolutionUpdate(Exception):
    '''Exception raised when trying to move an animal/particle to a bad location'''

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