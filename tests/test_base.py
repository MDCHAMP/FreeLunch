'''
Testing tech
'''
import pytest
import numpy as np

from freelunch.base import *
from freelunch.darwin import rand_1
from freelunch.benchmarks import exponential

def test_hyp_parse():
    opt = optimiser(exponential())
    assert(rand_1.__name__ == opt.parse_hyper(rand_1).__class__.__name__)

    with pytest.raises(AttributeError):
        opt.parse_hyper('Not a function')


def test_no_optimiser():
    with pytest.raises(TypeError):
        optimiser()


def test_naughty_obj():
    opt = optimiser(obj=lambda x: np.random.choice([np.nan, np.inf, 'a string']))
    for i in range(20):
        assert opt.obj([]) == None