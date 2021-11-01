'''
Testing tech
'''
import pytest
import numpy as np

from freelunch.base import *
from freelunch.darwin import rand_1

def test_hyp_parse():
    opt = optimiser()
    assert(rand_1.__name__ == opt.parse_hyper(rand_1).__class__.__name__)

    with pytest.raises(AttributeError):
        opt.parse_hyper('Not a function')


def test_no_optimiser():
    opt = optimiser()
    with pytest.raises(NotImplementedError):
        opt()
    with pytest.raises(NotImplementedError):
        opt.run()
    with pytest.raises(NotImplementedError):
        opt.run_quick()

def test_naughty_obj():
    opt = optimiser(obj=lambda x: np.random.choice([np.nan, np.inf, 'a string']))
    for i in range(20):
        assert opt.obj([]) == None