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


# @TODO more grnaular tests here

def test_early_stop():
    pass

def test_can_optimise():
    pass

