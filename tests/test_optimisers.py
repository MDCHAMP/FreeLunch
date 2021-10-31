'''
Testing the optimisers

Testing for function not performance see benchmarking script
'''
import pytest
import numpy as np
import json
np.random.seed(100)


from freelunch import DE, SA, PSO, SADE, KrillHerd, SA
from freelunch.benchmarks import exponential

optimiser_classes = [SA, DE, PSO, SADE, KrillHerd]
dims = [1,2,3]

def set_testing_hypers(opt):
    hypers = opt.hyper_defaults
    hypers['N'] = 5
    hypers['G'] = 2
    hypers['K'] = 2 # SA should really use G as well...
    
    return hypers


@pytest.mark.parametrize('opt', optimiser_classes)
def test_instancing_defaults(opt):
	o = opt()
	assert o.hypers == opt.hyper_defaults
    
# Since this happens in the base class it should be ok to just test DE
@pytest.mark.parametrize('n', [1,3,5])
def test_nfe(n):
    o = exponential(n)
    hypers = set_testing_hypers(DE)
    out = DE(obj=o, bounds=o.bounds, hypers=hypers)(nruns=n, full_output=True)
    assert out['nfe'] == (out['hypers']['G'] + 1) * out['hypers']['N'] * n 

@pytest.mark.parametrize('opt', optimiser_classes)
@pytest.mark.parametrize('n', [1,3])
@pytest.mark.parametrize('d', [1,3,5])
def test_run(opt,n,d):
    o = exponential(d)
    hypers = set_testing_hypers(opt)
    out = opt(obj=o, bounds=o.bounds, hypers=hypers)(nruns=n, full_output=True)
    assert(len(out['solutions'])==n*hypers['N'])
    assert all(x<=y for x, y in zip(out['scores'], out['scores'][1:])) # scores are ordered


@pytest.mark.parametrize('opt', optimiser_classes)
@pytest.mark.parametrize('d', [1,3,5])
def test_run_one(opt,n,d):
    o = exponential(d)
    hypers = set_testing_hypers(opt)
    out = opt(obj=o, bounds=o.bounds, hypers=hypers)(full_output=True)
    assert(len(out['solutions'])==hypers['N'])


@pytest.mark.parametrize('opt', optimiser_classes)
@pytest.mark.parametrize('n', [1,3])
@pytest.mark.parametrize('m', [1,3])
@pytest.mark.parametrize('d', [1,5])
def test_run_not_full(opt,n,m,d):
    o = exponential(d)
    hypers = set_testing_hypers(opt)
    out = opt(obj=o, bounds=o.bounds, hypers=hypers)(nruns=n, return_m=m, full_output=False)
    assert(len(out)==m)
    assert([(len(s) == d) for s in out])

@pytest.mark.parametrize('opt', optimiser_classes)
@pytest.mark.parametrize('n', [1,3])
def test_can_json(opt,n):
    o = exponential(1)
    hypers = set_testing_hypers(opt)
    out = opt(obj=o, bounds=o.bounds, hypers=hypers)(nruns=n, full_output=True)
    s = json.dumps(out)