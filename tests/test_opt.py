'''
Testing 
'''
import pytest
import numpy as np
np.random.seed(100)

from freelunch import DE, SA, PSO, SADE, KrillHerd, SA
from freelunch.benchmarks import ackley, exponential, happycat, periodic

optimiser_classes = [SA, DE, PSO, SADE, KrillHerd]
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
	out = opt(obj=o, bounds=o.bounds)(full_output=True)
	assert type(out) is dict # returns dict
	assert all(x<=y for x, y in zip(out['scores'], out['scores'][1:])) # scores are ordered
	assert (obj.tol is None) or (abs(out['scores'][0] - obj.f0) < obj.tol) # within tolerance


# Do you even lift bro?

@pytest.mark.benchmark(group="Optimisers")
@pytest.mark.parametrize('opt', optimiser_classes)
def test_bench_optimisers(benchmark, opt):
	o = exponential(1)
	bench_opt = opt(obj=o, bounds=o.bounds)
	benchmark(bench_opt, return_m=1)

@pytest.mark.benchmark(group="Benchmarks")
@pytest.mark.parametrize('obj', benchmark_problems)
def test_bench_benchmarks(benchmark, obj):
	o = obj(2)
	optima = benchmark(o, o.optimum)
	assert  optima == o.f0
