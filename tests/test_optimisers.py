"""
Testing the optimisers

Testing for function not performance see benchmarking script
"""


import inspect
import json

import numpy as np
import pytest

from freelunch import optimisers, tech
from freelunch.benchmarks import exponential

# %% Fixtures

np.random.seed(100)

# Pull out all optimisers automatically - sorry Max!
optimiser_classes = [
    cl
    for name, cl in inspect.getmembers(optimisers)
    if inspect.isclass(cl)
    and name != "optimiser"
    and issubclass(cl, optimisers.optimiser)
]
dims = [1, 2, 3]


def early_stopper(state):
    if state.gen == 3:
        return False

def set_testing_hypers(opt):
    hypers = opt.hyper_defaults
    hypers["N"] = 10
    hypers["G"] = 11
    return hypers


# %% Test optmisers


@pytest.mark.parametrize("opt", optimiser_classes)
def test_instancing_defaults(opt):
    o = opt(exponential())
    for k, v in o.hypers.items():
        if k in opt.hyper_defaults:
            assert np.all(v == opt.hyper_defaults[k])


@pytest.mark.parametrize("opt", optimiser_classes)
@pytest.mark.parametrize("n", [1, 2])
def test_can_json(opt, n):
    o = exponential(2)
    hypers = set_testing_hypers(opt)
    best, runs = opt(obj=o, bounds=o.bounds, hypers=hypers)(n_runs=n)
    s = json.dumps(runs, cls=tech.freelunch_json_encoder)

def test_early_stop():
    o = exponential(2)
    opt = optimisers.DE(o, o.bounds)
    opt.post_step_hook = early_stopper
    opt()
    assert opt.gen == 3

@pytest.mark.parametrize("n", [1,2])
def test_can_mp(n):
    obj = exponential(2)
    opt = optimisers.DE(obj, obj.bounds, {'N':10, 'G':10})
    opt(1, n)

@pytest.mark.parametrize("opt", optimiser_classes)
def test_can_optimise(opt): # this is a flaky test but wcyd
    obj = exponential(1)
    optim = opt(obj, obj.bounds)
    (x0, f0), runs = optim()
    assert f0 < -0.9
