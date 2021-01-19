'''
Testing 
'''
import pytest
import numpy as np

from freelunch import DE, SA, PSO, KrillHerd
from freelunch.benchmarks import ackley, exponential, happycat, periodic

optimiser_classes = [DE, PSO, KrillHerd]
benchmark_problems = [ackley, exponential, happycat, periodic]
dims = [1,2,3,4]

@pytest.mark.parametrize('opt', optimiser_classes)
def test_instancing_defaults(opt):
	o = opt()
	assert o.hypers == opt.hyper_defaults

@pytest.mark.parametrize('obj', benchmark_problems)
@pytest.mark.parametrize('n', dims)
def test_true_optima(obj, n):
	b = obj(n)
	evaluates = b(b.optimum)
	assert evaluates == b.f0

@pytest.mark.parametrize('opt', optimiser_classes)
@pytest.mark.parametrize('obj', benchmark_problems)
@pytest.mark.parametrize('n', dims)
def test_run_opt_with_defualts(opt, obj, n):
	o = obj(n)
	np.random.seed(1)
	out = opt(obj=o, bounds=o.bounds)(full_output=True)
	assert type(out) is dict # returns dict
	assert all(x<=y for x, y in zip(out['scores'], out['scores'][1:])) # scores are ordered
	assert (obj.tol is None) or (abs(out['scores'][0] - obj.f0) < obj.tol) # within tolerance
