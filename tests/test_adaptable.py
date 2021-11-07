'''
Testing adaptables
'''
import pytest
import numpy as np

np.random.seed(100)

from freelunch.adaptable import *
from freelunch.optimisers import SADE
from freelunch.benchmarks import ackley

# Methods

def test_adaptable_method():
    with pytest.raises(NotImplementedError):
        adaptable_method()() 
    
    m = adaptable_method()
    m.op = lambda: None
    for lp in range (5):
        for i in range(10):
            m()
            m.win()
        assert m.hits[-1] == 10
        assert m.wins[-1] == 10 
        m.reset_counts()
        assert m.hits[-1] == 0
        assert m.wins[-1] == 0


def test_adaptable_set():
    # TODO proper testing for now just run SADE with low LP
    opt = SADE(ackley(1), bounds=[[-1, 1]])
    opt.hypers['N'] = 4
    opt.hypers['G'] = 12
    opt.hypers['Lp'] = 2
    opt()


# Parmaters

def test_base_parameter():
    p = adaptable_parameter(100)
    assert p() == 100


def test_lin_varying():
    p = linearly_varying_parameter(0, 1, 100)
    t = np.linspace(0, 1, 100)
    assert np.all([p(k)==t[k] for k in range(100)])
    p = linearly_varying_parameter(1, 1, 100)
    assert np.all([p(k)==1 for k in range(100)])

def test_normally_varying():
    p = normally_varying_parameter(0,1)
    for lp in range (5):
        for i in range(10):
            assert p() < 10 # sanity
            p.win(1)
        assert p.hits[-1] == 10
        assert p.wins[-1] == 10 
        assert p.win_values == [1]*10
        p.update()
        assert p.hits[-1] == 0
        assert p.wins[-1] == 0
        assert p.win_values == []

    p.win_values = np.array([np.nan, np.nan])
    with pytest.raises(ValueError):
        p.update()

    