'''
Testing the optimisers

Testing for function not performance see benchmarking script
'''
from freelunch.benchmarks import exponential
from freelunch import optimisers
from itertools import permutations
import inspect
import pytest
import numpy as np
import json
np.random.seed(200)

# Pull out all optimisers automatically - sorry Max!
optimiser_classes = [
    cl for name, cl in inspect.getmembers(optimisers) 
    if inspect.isclass(cl) and name != "optimiser" and issubclass(cl,optimisers.optimiser)]
dims = [1, 2, 3]


def set_testing_hypers(opt):
    hypers = opt.hyper_defaults
    hypers['N'] = 10
    hypers['G'] = 2
    hypers['K'] = 2  # SA should really use G as well...
    return hypers

# @pytest.mark.parametrize('n', np.arange(5)+1)
# def test_full_output(n):
#     hypers = set_testing_hypers(DE)
#     opt = DE(obj=lambda x: 1, hypers=hypers, bounds=[[-1, 1]]*n)
#     opt(n_runs=1, full_output=True)

@pytest.mark.parametrize('opt', optimiser_classes)
def test_instancing_defaults(opt):
    o = opt(exponential())
    for k, v in o.hypers.items():
        if k in opt.hyper_defaults:
            assert np.all(v == opt.hyper_defaults[k])

# Since this happens in the base class it should be ok to just test DE
# @pytest.mark.parametrize('n', [1, 3, 5])
# def test_nfe(n):
#     o = exponential(n)
#     hypers = set_testing_hypers(DE)
#     out = DE(obj=o, bounds=o.bounds, hypers=hypers)(n_runs=n, full_output=True)
#     assert out['nfe'] == (out['hypers']['G'] + 1) * out['hypers']['N'] * n

# @pytest.mark.parametrize('n', [1, 3, 5])
# def test_multiproc(n):
#     o = exponential(2)
#     hypers = set_testing_hypers(DE)
#     out = DE(obj=o, bounds=o.bounds, hypers=hypers)(n_runs=n, full_output=True, n_workers=n)
#     assert out['nfe'] == (out['hypers']['G'] + 1) * out['hypers']['N'] * n

# Test every optimiser to catch pickling bugs - you never do know...
# Multiprocessing on the back burner
# @pytest.mark.parametrize('opt', optimiser_classes)
# def test_multiproc_optimisers(opt):
#     o = exponential(2)
#     hypers = set_testing_hypers(opt)
#     opt(obj=o, bounds=o.bounds, hypers=hypers)(n_runs=4, full_output=True, n_workers=2)


@pytest.mark.parametrize('opt', optimiser_classes)
@pytest.mark.parametrize('n', [1, 3])
@pytest.mark.parametrize('d', [1, 3, 5])
def test_run(opt, n, d):
    np.random.seed(200)
    o = exponential(d)
    hypers = set_testing_hypers(opt)
    out = opt(obj=o, bounds=o.bounds, hypers=hypers)(n_runs=n, full_output=True)
    assert(len(out['solutions']) == n*hypers['N'])
    # scores are ordered
    assert(all(x <= y for x, y in zip(out['scores'], out['scores'][1:])))
    for o in out['solutions']:
        for i, v in enumerate(o):
            assert(v > out['bounds'][i][0])
            assert(v < out['bounds'][i][1])


@pytest.mark.parametrize('opt', optimiser_classes)
@pytest.mark.parametrize('d', [1, 3, 5])
def test_run_one(opt, d):
    o = exponential(d)
    hypers = set_testing_hypers(opt)
    out = opt(obj=o, bounds=o.bounds, hypers=hypers)()
    # assert(len(out['solutions']) == hypers['N'])
    assert len(out) == 2
    assert isinstance(out[0], float)
    assert isinstance(out[1], np.ndarray)
    assert out[1].shape[0] == o.bounds.shape[0]


@pytest.mark.parametrize('opt', optimiser_classes)
@pytest.mark.parametrize('n', [1, 3])
@pytest.mark.parametrize('m', [1, 3])
@pytest.mark.parametrize('d', [1, 5])
def test_run_not_full(opt, n, m, d):
    o = exponential(d)
    hypers = set_testing_hypers(opt)
    out = opt(obj=o, bounds=o.bounds, hypers=hypers)(
        n_runs=n, n_return=m, full_output=False)
    assert(len(out) == m)
    assert([(len(s) == d) for s in out])


@pytest.mark.parametrize('opt', optimiser_classes)
@pytest.mark.parametrize('n', [1, 3])
def test_can_json(opt, n):
    o = exponential(1)
    hypers = set_testing_hypers(opt)
    out = opt(obj=o, bounds=o.bounds, hypers=hypers)(n_runs=n, full_output=True)
    s = json.dumps(out)


@pytest.mark.parametrize('opt', optimiser_classes)
def test_repr(opt):
    rep = opt(exponential()).__repr__()
    assert(rep == (opt.name + ' optimisation object'))

@pytest.mark.parametrize('d', [1, 2])
def test_abc_crossover(d):
    """Test the stupid crossover thing in ABC

    Args:
        d (int): Dimensionality
    """
    N = 5
    o = exponential(d)
    hypers = set_testing_hypers(optimisers.ABC)
    opt = optimisers.ABC(obj=o, bounds=o.bounds, hypers=hypers)
    # Fake a run
    pos = np.random.standard_normal((N,d))
    crossovers = opt.crossover_points(pos)
    all_possible = []
    for a,b in permutations(pos, 2):
        all_possible.append(a-b)
    all_possible = np.array(all_possible)

    for c in crossovers:
        assert np.any(c == all_possible)
    



