'''
Testing tech
'''
from freelunch.zoo import animal, particle, krill
from freelunch.tech import *
import pytest
import numpy as np

np.random.seed(100)


animals = [None, animal, particle, krill]


def dummy_obj_fun(dna):
    return 1


def bound_fixture(a, b, bnds, bounder, **hypers):
    a = animal(dna=a)
    bounder(a, bnds, **hypers)
    assert np.all(a.dna == b)


@pytest.mark.parametrize('N', [1, 3])
@pytest.mark.parametrize('dim', [1, 3])
@pytest.mark.parametrize('creature', animals)
def test_uniform_init(N, dim, creature):

    bounds = np.empty((dim, 2))
    for n in range(dim):
        bounds[n, 1] = 10*np.random.rand()
        bounds[n, 0] = -bounds[n, 1]

    if creature is None:
        pop = uniform_continuous_init(bounds, N)
    else:
        pop = uniform_continuous_init(bounds, N, creature=creature)

    assert(len(pop) == N)
    assert(np.all(pop[0].dna > bounds[:, 0]))
    assert(np.all(pop[0].dna < bounds[:, 1]))

@pytest.mark.parametrize('N', [1, 3])
@pytest.mark.parametrize('dim', [1, 3])
@pytest.mark.parametrize('creature', animals)
def test_gaussian_init(N, dim, creature):

    bounds = np.empty((dim, 2))
    for n in range(dim):
        bounds[n, 1] = 10*np.random.rand()
        bounds[n, 0] = -bounds[n, 1]

        mu = np.random.uniform(-bounds[n, 1]/5, bounds[n, 1]/5)
        sig = np.random.uniform(0,1)
    
    if creature is None:
        pop = Gaussian_neigbourhood_init(bounds, N, mu=mu, sig=sig)
    else:
        pop = Gaussian_neigbourhood_init(bounds, N, creature=creature)
        assert(np.all(pop[0].dna > bounds[:, 0]))
        assert(np.all(pop[0].dna < bounds[:, 1]))
    assert(len(pop) == N)




def test_compute_obj():

    N = 10
    pop = uniform_continuous_init(np.array([[-1., 1.]]), N)
    pop = compute_obj(pop, dummy_obj_fun)

    assert(len(pop) == N)
    for p in pop:
        assert(p.fitness == 1)


def test_sticky_bounds():
    bounds = [[-1, 1], [-2, 2]]
    eps = 1e-12
    bound_fixture([0.5, 1.2], [0.5, 1.2], bounds, sticky_bounds, eps=eps)
    bound_fixture([-2, 1.2],  [-1+eps, 1.2],bounds, sticky_bounds, eps=eps)
    bound_fixture([0.5, 2.2], [0.5, 2-eps],bounds, sticky_bounds, eps=eps)
    bound_fixture([-1.5, 2.2], [-1+eps, 2-eps],bounds, sticky_bounds, eps=eps)
    bound_fixture([1.5, -2.2], [1-eps, -2+eps],bounds, sticky_bounds, eps=eps)



def test_no_bounds():
    with pytest.raises(Warning):
        no_bounding(animal(), [])


def test_bounds_as_mat():

    bounds = [[-1, 1], [-2, 2]]
    bounds_mat = np.array(bounds)
    assert(np.all(bounds_as_mat(bounds) == bounds_mat))


def test_lin_reduce():

    lims = [2, 4]
    n_max = 10
    assert(lin_reduce(lims, 0, n_max) == 4)
    assert(lin_reduce(lims, n_max, n_max) == 2)
    assert(lin_reduce(lims, 5, n_max) == 3)

    lims = [4, 2]
    n_max = 10
    assert(lin_reduce(lims, 0, n_max) == 4)
    assert(lin_reduce(lims, n_max, n_max) == 2)
    assert(lin_reduce(lims, 5, n_max) == 3)


def test_pdist():

    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    DAA = np.empty((2, 2))
    DAB = np.empty((2, 2))

    # Slow lame euclidean distance to check big brain
    for i in range(2):
        for j in range(2):
            DAA[i, j] = np.sqrt(np.sum((A[i, :] - A[j, :])**2))
            DAB[i, j] = np.sqrt(np.sum((A[i, :] - B[j, :])**2))

    assert(np.all(pdist(A) == DAA))
    assert(np.all(pdist(A, A) == DAA))
    assert(np.all(pdist(A, B) == DAB))
