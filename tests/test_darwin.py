from numpy.lib.type_check import real
import pytest
import numpy as np

from freelunch.zoo import animal, krill

np.random.seed(100)

from freelunch.darwin import *
from freelunch.tech import compute_obj, uniform_continuous_init
from freelunch.benchmarks import exponential
from freelunch.util import real_finite

search_ops = [rand_1, rand_2, best_1, best_2, current_1]
cross_ops = [binary_crossover]
select_ops = [binary_tournament]

@pytest.mark.parametrize('D',[1,3])
@pytest.mark.parametrize('op', search_ops)
def test_search(op,D):


    bounds = np.repeat(np.array([[-1,1]]),D,axis=0)
    N = 10
    obj = exponential(N)

    pop = uniform_continuous_init(bounds,N)
    pop = compute_obj(pop, obj)

    dna_moved = op().op(pop[0],pop=pop)
    dna_moved2 = op().op(pop[0],pop=pop, F=0.2)

    assert([real_finite(d) for d in dna_moved])
    assert([real_finite(d) for d in dna_moved2])
    assert(np.all(pop[0].dna.shape == dna_moved.shape))
    assert(np.all(pop[0].dna.shape == dna_moved2.shape))
    # Are these allowed to generate out of bounds proposals?
    # assert(np.all(dna_moved>bounds[:,0]) and np.all(dna_moved<bounds[:,1]))

@pytest.mark.parametrize('D',[1,3])
@pytest.mark.parametrize('op', cross_ops)
def test_cross(op,D):


    bounds = np.repeat(np.array([[-1,1]]),D,axis=0)
    N = 10
    obj = exponential(N)

    pop = uniform_continuous_init(bounds,N)
    pop = compute_obj(pop, obj)

    dna_moved = op().op(pop[0].dna,pop[1].dna)
    dna_moved2 = op().op(pop[0].dna,pop[1].dna, Cr=0.2)

    assert([real_finite(d) for d in dna_moved])
    assert([real_finite(d) for d in dna_moved2])
    assert(np.all(pop[0].dna.shape == dna_moved.shape))
    assert(np.all(pop[0].dna.shape == dna_moved2.shape))
    # Are these allowed to generate out of bounds proposals?
    # assert(np.all(dna_moved>bounds[:,0]) and np.all(dna_moved<bounds[:,1]))

@pytest.mark.parametrize('op',select_ops)
def test_selection(op):

    D = 3
    bounds = np.repeat(np.array([[-1,1]]),D,axis=0)
    N = 10
    obj = exponential(N)

    pop_old = uniform_continuous_init(bounds,N)
    pop_old = compute_obj(pop_old, obj)

    pop_new = uniform_continuous_init(bounds,N)
    pop_new = compute_obj(pop_new, obj)

    pop = op().op(pop_old, pop_new)

    
    for p, o, n in zip(pop, pop_old, pop_new):
        # Testing for fitness equality
        assert(p.fitness == o.fitness or p.fitness == n.fitness )
        # Testing for complete equality
        assert(p == o or p == n)

