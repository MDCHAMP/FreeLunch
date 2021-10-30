'''
Testing 
'''
import pytest
import numpy as np
import json
np.random.seed(100)


from freelunch import DE, SA, PSO, SADE, KrillHerd, SA
from freelunch.base import optimiser
from freelunch.benchmarks import ackley, exponential, happycat, periodic

optimiser_classes = [SA, DE, PSO, SADE, KrillHerd]
benchmark_problems = [ackley, exponential, happycat, periodic]
dims = [1,2,3]

def naughty_objective(dna):
	return np.random.choice([np.nan, np.inf, None, dna, 'im not a valid score'])


def test_base_optimiser():
	opt = optimiser()
	with pytest.raises(NotImplementedError):
		opt()
	opt.can_run_quick = True
	opt.obj = lambda x:x
	opt(nruns=2)

def test_naughty_objective():
	out = DE(obj=naughty_objective, bounds=[[-1, 1]])(nruns=1, full_output=True)

@pytest.mark.parametrize('opt', optimiser_classes)
def test_instancing_defaults(opt):
	o = opt()
	assert o.hypers == opt.hyper_defaults


@pytest.mark.parametrize('n', [1,3,5])
def test_nfe(n):
	o = ackley(1)
	out = DE(obj=o, bounds=o.bounds)(nruns=n, full_output=True)
	assert out['nfe'] == (out['hypers']['G'] + 1) * out['hypers']['N'] * n 

@pytest.mark.parametrize('n', [1,3,5])
def test_mruns(n):
	o = ackley(1)
	out = DE(obj=o, bounds=o.bounds)(nruns=n, full_output=True)
	assert type(out) is dict # returns dict
	assert all(x<=y for x, y in zip(out['scores'], out['scores'][1:])) # scores are ordered

@pytest.mark.parametrize('n', [1,5])
def test_can_json(n):
	o = ackley(1)
	out = DE(obj=o, bounds=o.bounds)(nruns=n, full_output=True)
	s = json.dumps(out)
 
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
