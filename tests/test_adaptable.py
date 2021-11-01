'''
Testing adaptables
'''
import pytest
import numpy as np

np.random.seed(100)

from freelunch.adaptable import *


def test_adaptable_method():
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
    a = adaptable_set()
    # TODO proper testing


def test_lin_varying():
    p = linearly_varying_parameter(0, 1, 100)
    t = np.linspace(0, 1, 100)
    assert np.all([p(k)==t[k] for k in range(100)])


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