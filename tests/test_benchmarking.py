from freelunch.benchmarks import ackley, exponential, happycat, periodic
from freelunch.base import optimiser
from freelunch import DE, SA, PSO, PAO, SADE, KrillHerd, SA, QPSO
import pytest
import numpy as np
import json
np.random.seed(100)

optimiser_classes = [SA, DE, PSO, PAO, SADE, KrillHerd, QPSO]
benchmark_problems = [ackley, exponential, happycat, periodic]

@pytest.mark.benchmark(group="Optimisers")
@pytest.mark.parametrize('opt', optimiser_classes)
def test_bench_optimisers(benchmark, opt):
    o = exponential(1)
    bench_opt = opt(obj=o, bounds=o.bounds)
    benchmark(bench_opt) 

@pytest.mark.benchmark(group="Benchmarks")
@pytest.mark.parametrize('obj', benchmark_problems)
def test_bench_benchmarks(benchmark, obj):
    o = obj(2)
    optima = benchmark(o, o.optimum)
    assert optima == o.f0
