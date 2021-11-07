'''
Testing tech
'''
import pytest
import numpy as np

from freelunch.base import *
from freelunch.darwin import rand_1
from freelunch.tech import Bounder

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

@pytest.mark.parametrize('dim',[1,3])
def test_bounding(dim):

    bounds = np.empty((dim,2))
    for n in range(dim):
        bounds[n,1] = 10*np.random.rand()
        bounds[n,0] = -bounds[n,1]

    opt = optimiser(bounds=bounds)
    assert(np.all(bounds == opt.bounds.bounds ))

    B = Bounder(bounds)
    opt = optimiser(bounds=B)
    assert(np.all(bounds == opt.bounds.bounds ))

    B = ('Bounder', bounds)
    opt = optimiser(bounds=B)
    assert(np.all(bounds == opt.bounds.bounds ))

    hypers = {'eps':1e-10}
    B = ('Bounder', bounds, hypers)
    opt = optimiser(bounds=B)
    assert(np.all(bounds == opt.bounds.bounds ))

    with pytest.raises(ValueError):
        opt = optimiser(bounds=1)
    
    with pytest.raises(ValueError):
        opt = optimiser(bounds=(1,))
    
    with pytest.raises(ValueError):
        opt = optimiser(bounds=(1,1,1,1))

    opt = optimiser(bounds=[[-1,1]])
    assert(np.all([[-1,1]] == opt.bounds.bounds ))



