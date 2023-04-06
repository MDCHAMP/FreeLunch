# '''
# Testing tech
# '''

from freelunch.tech import *
import pytest
import numpy as np

np.random.seed(100)

animals = [None]


def dummy_obj_fun(dna):
    return 1


def bound_fixture(a, b, bnds, bounder, **hypers):
    class dummy_opt():
        def __init__(self, x, bounds):
            self.pos = x
            self.bounds = bounds
    opt = dummy_opt(np.tile(a,[10,1]), np.array(bnds)) # Make a population array
    bounder(opt)
    assert np.allclose(opt.pos, b)


def test_greedy_selection():

    fit_old = np.array([1,2,3])
    fit_new = np.array([2,0,4])
    idx = fit_new < fit_old

    pos_old = np.ones((3,2))
    pos_old_copy = pos_old.copy()
    pos_new = 3*np.ones((3,2))

    vel_old = -1*np.ones((3,2))
    vel_old_copy = vel_old.copy()
    vel_new = 2*np.ones((3,2))

    greedy_selection(fit_old, fit_new, pos_old, pos_new)

    assert np.all(pos_old[idx] == pos_new[idx])
    assert np.all(pos_old[~idx] == pos_old_copy[~idx])

    fit_old = np.array([1,2,3])
    fit_new = np.array([2,0,4])
    idx = fit_new < fit_old

    pos_old = np.ones((3,2))
    pos_old_copy = pos_old.copy()
    pos_new = 3*np.ones((3,2))


    greedy_selection(fit_old, fit_new, [pos_old], [pos_new])

    assert np.all(pos_old[idx] == pos_new[idx])
    assert np.all(pos_old[~idx] == pos_old_copy[~idx])

    fit_old = np.array([1,2,3])
    fit_new = np.array([2,0,4])
    idx = fit_new < fit_old

    pos_old = np.ones((3,2))
    pos_old_copy = pos_old.copy()
    pos_new = 3*np.ones((3,2))

    vel_old = -1*np.ones((3,2))
    vel_old_copy = vel_old.copy()
    vel_new = 2*np.ones((3,2))

    
    greedy_selection(fit_old, fit_new, [pos_old, vel_old], [pos_new, vel_new])

    assert np.all(pos_old[idx] == pos_new[idx])
    assert np.all(pos_old[~idx] == pos_old_copy[~idx])
    assert np.all(vel_old[idx] == vel_new[idx])
    assert np.all(vel_old[~idx] == vel_old_copy[~idx])
    


@pytest.mark.parametrize('N', [1, 3])
@pytest.mark.parametrize('dim', [1, 3])
def test_uniform_init(N, dim):

    bounds = np.empty((dim, 2))
    for n in range(dim):
        bounds[n, 1] = 10*np.random.rand()
        bounds[n, 0] = -bounds[n, 1]

    pop = uniform_continuous_init(bounds, N)

    assert(len(pop) == N)
    assert(np.all(pop > bounds[:, 0]))
    assert(np.all(pop < bounds[:, 1]))

@pytest.mark.parametrize('N', [1, 3])
@pytest.mark.parametrize('dim', [1, 3])
def test_gaussian_init(N, dim):

    if dim == 1:
        bounds = np.empty((dim, 2))
        for n in range(dim):
            bounds[n, 1] = 10*np.random.rand()
            bounds[n, 0] = -bounds[n, 1]
            
            mu = np.random.uniform(-bounds[n, 1]/5, bounds[n, 1]/5)
            sig= np.random.uniform(0,1)

        pop = gaussian_neigbourhood_init(bounds, N, mu=mu, sig=sig)

        assert(pop.shape[0] == N)
        assert(pop.shape[1] == dim)
        assert(np.all(pop > 5*bounds[:, 0]))
        assert(np.all(pop < 5*bounds[:, 1]))


    bounds = np.empty((dim, 2))
    mu = []
    sig = []
    for n in range(dim):
        bounds[n, 1] = 10*np.random.rand()
        bounds[n, 0] = -bounds[n, 1]
        
        mu.append(np.random.uniform(-bounds[n, 1]/5, bounds[n, 1]/5))
        sig.append(np.random.uniform(0,1))
    
    mu = np.array(mu)
    sig = np.array(sig)    

    pop = gaussian_neigbourhood_init(bounds, N, mu=mu, sig=sig)

    assert(pop.shape[0] == N)
    assert(pop.shape[1] == dim)
    assert(np.all(pop > 5*bounds[:, 0]))
    assert(np.all(pop < 5*bounds[:, 1]))

    pop = gaussian_neigbourhood_init(bounds, N)

    assert(pop.shape[0] == N)
    assert(pop.shape[1] == dim)
    assert(np.all(pop > 5*bounds[:, 0]))
    assert(np.all(pop < 5*bounds[:, 1]))


# TODO: Is this needed, I don't think so...
# def test_compute_obj():

#     N = 10
#     pop = uniform_continuous_init(np.array([[-1., 1.]]), N)
#     pop = compute_obj(pop, dummy_obj_fun)

#     assert(len(pop) == N)
#     for p in pop:
#         assert(p.fitness == 1)


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
        no_bounding(None)


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
