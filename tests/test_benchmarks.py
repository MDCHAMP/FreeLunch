'''
Testing benchmark problems
'''
import pytest
import numpy as np

np.random.seed(100)

from freelunch.util import ZeroLengthSolutionError
from freelunch.benchmarks import ackley, exponential, happycat, periodic

benchmark_problems = [ackley, exponential, happycat, periodic]
dims = [1,2,3]

@pytest.mark.parametrize('obj', benchmark_problems)
@pytest.mark.parametrize('n', dims)
def test_true_optima(obj, n):
	b = obj(n)
	evaluates = b(b.optimum)
	assert evaluates == b.f0

@pytest.mark.parametrize('obj', benchmark_problems)
def test_err(obj):

    with pytest.raises(ZeroLengthSolutionError):
        obj()([])

