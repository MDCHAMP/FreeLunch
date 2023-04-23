import json

import numpy as np
import pytest

from freelunch.tech import *


# %% Fixtures

test_pos = np.random.uniform(size=(100, 2))
test_fit = np.random.uniform(size=(100,))
MUTATORS = [rand_1, rand_2, best_1, best_2, rand_to_best_1]


def bound_fixt(pos, bounder, bnds, eps=1e-12):
    pos = bounder(np.array(pos), bnds)
    assert np.all(pos > (bnds[:, 0] - eps))
    assert np.all(pos < (bnds[:, 1] + eps))


# %% Intialisation strategies


@pytest.mark.parametrize("N", [1, 3])
@pytest.mark.parametrize("dim", [1, 3])
def test_uniform_init(N, dim):
    bounds = np.empty((dim, 2))
    for n in range(dim):
        bounds[n, 1] = 10 * np.random.rand()
        bounds[n, 0] = -bounds[n, 1]

    pop = uniform_continuous_init(bounds, N)

    assert len(pop) == N
    assert np.all(pop > bounds[:, 0])
    assert np.all(pop < bounds[:, 1])


# %% Mutation strategies


@pytest.mark.parametrize("M", MUTATORS)
@pytest.mark.parametrize("k", [None, 1, np.arange(10)])
@pytest.mark.parametrize("F", [None, 1])
def test_mutators(M, k, F):
    if F is None:
        F = np.ones(len(test_pos))
    res = M((test_pos.copy(), test_fit.copy()), 0.5, k)
    if k is None:
        lk = len(test_pos)
    elif type(k) is not int:
        lk = len(k)
    else:
        lk = 1
    assert res.shape == (lk, test_pos.shape[1])


# %% Crossover strategies


@pytest.mark.parametrize("Cr", [0.1, 1, np.arange(100)[:, None]])
def test_binary_crossover(Cr):
    p1 = test_pos
    p2 = np.random.uniform(size=(100, 2))
    p3 = binary_crossover(p1.copy(), p2.copy(), Cr, jrand=True)
    assert not np.all(p3 == p1)
    p4 = binary_crossover(p1.copy(), p1.copy(), Cr, jrand=False)
    assert np.all(p4 == p1)


# %% Selection strategies


def test_greedy_selection():
    p1 = test_pos.copy(), np.ones(len(test_pos))
    p2 = test_pos[:] + 1, np.zeros(len(test_pos))
    p3 = greedy_selection(p1, p2, return_idx=False)
    assert np.all(p3[0] == test_pos + 1)
    p1 = np.random.uniform(size=(100, 2)), np.random.uniform(size=(100,))
    p2 = np.random.uniform(size=(100, 2)), np.random.uniform(size=(100,))
    p3, idx = greedy_selection(p1, p2, return_idx=True)
    assert np.all(p3[0][idx] == p2[0][idx])
    assert np.all(p3[1][idx] == p2[1][idx])
    assert np.all(p3[0][~idx] == p1[0][~idx])
    assert np.all(p3[1][~idx] == p1[1][~idx])
    assert not np.all(p3[0] == p1[0])


def test_global_best():
    b1 = [], 100
    p1 = np.random.uniform(size=(100, 2)), np.random.uniform(size=(100,))
    b2 = update_best(b1, p1)
    assert b2 != b1
    b1 = [], -1
    p1 = np.random.uniform(size=(100, 2)), np.random.uniform(size=(100,))
    b2 = update_best(b1, p1)
    assert b2 == b1


# %% Bounding Strategies


def test_sticky_bounds():
    bounds = np.array([[-1, 1], [0, 1]])
    pos = test_pos
    bound_fixt(pos, sticky_bounds, bounds)
    pos[0, 0] = -2
    bound_fixt(pos, sticky_bounds, bounds)
    pos[0, 1] = -0.1
    bound_fixt(pos, sticky_bounds, bounds)
    pos[-1, -1] = np.inf
    bound_fixt(pos, sticky_bounds, bounds)


def test_no_bounds():
    with pytest.warns(Warning):
        no_bounding(test_pos, None)


# %% Adaptable parameters


def test_lin_reduce():
    lims = [2, 4]
    n_max = 10
    assert lin_vary(lims, 0, n_max) == 4
    assert lin_vary(lims, n_max, n_max) == 2
    assert lin_vary(lims, 5, n_max) == 3
    lims = [4, 2]
    n_max = 10
    assert lin_vary(lims, 0, n_max) == 2
    assert lin_vary(lims, n_max, n_max) == 4
    assert lin_vary(lims, 5, n_max) == 3


@pytest.mark.parametrize("S", [[], [2, 2, 2], [1, 2, 3]])
def test_normal_update(S):
    p1 = 2, 1
    p2 = normal_update(p1, S)
    assert p2[0] == 2
    assert np.isfinite(p2[1])
    assert p2[1] > 0


def test_update_selection_probs():
    hits = np.array([1, 2, 3, 5])
    wins = np.array([1, 0, 2, 1])
    ps = update_selection_probs(hits, wins)
    assert sum(ps) == 1
    assert np.all(ps > 0)


def test_track_hits_wins():
    hits = np.array([0, 0])
    wins = np.array([0, 0])
    scores = hits, wins
    selection_idx = np.array([0, 1, 0, 1, 0])
    success_idx = np.array([0, 0, 0, 1, 1])
    hits, wins = track_hits_wins(scores, selection_idx, success_idx)
    assert np.all(hits == np.array([3, 2]))
    assert np.all(wins == np.array([1, 1]))


# %% Misc testing


@pytest.mark.parametrize("N", [1, 10, 100])
@pytest.mark.parametrize(
    "n",
    [
        1,
        3,
        5,
    ],
)
@pytest.mark.parametrize("k", [None, 1, np.arange(5)])
def test_parent_idx_no_repeats(N, n, k):
    if k is None:
        lk = N
    elif type(k) is not int:
        lk = len(k)
    else:
        lk = 1
    if N <= n:
        with pytest.raises(AssertionError):
            parent_idx_no_repeats(N, n, k)
        return
    else:
        idxs = parent_idx_no_repeats(N, n, k)
        assert idxs.shape == (n, lk)
        for row in idxs.T:
            assert len(row) == n
            assert len(np.unique(row)) == n


def test_pdist():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    DAA = np.empty((2, 2))
    DAB = np.empty((2, 2))
    # Slow lame euclidean distance to check big brain
    for i in range(2):
        for j in range(2):
            DAA[i, j] = np.sqrt(np.sum((A[i, :] - A[j, :]) ** 2))
            DAB[i, j] = np.sqrt(np.sum((A[i, :] - B[j, :]) ** 2))

    assert np.all(pdist(A) == DAA)
    assert np.all(pdist(A, A) == DAA)
    assert np.all(pdist(A, B) == DAB)


def test_expm():
    z = np.zeros((2, 2))
    assert np.allclose(expm(z), np.eye(2))
    d = np.eye(2)
    assert np.allclose(expm(d), np.diag(np.exp(np.diag(d))))
    a = np.array([[2, 0], [0, 3]])
    b = np.array([[1, 0], [0, 8]])
    assert np.allclose(expm(a + b), expm(a) @ expm(b))
    assert np.allclose(expm(a.T), expm(a).T)


def test_json_encoder():
    json.dumps(
        {"arr": np.array([1, 2, 3]), "list": [1, 2, 3]}, cls=freelunch_json_encoder
    )
