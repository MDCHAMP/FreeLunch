import pytest
import numpy as np

np.random.seed(0)

import freelunch
from freelunch.base import *

obj = lambda x: 0


def test_no_obj():
    with pytest.raises(TypeError):
        optimiser()


def test_no_bounds():
    opt = optimiser(obj)
    with pytest.raises(Warning):
        opt.bounder(None, None)


def test_bounds():
    opt = optimiser(obj, bounds=[[-1, 1]])
    assert opt.bounder is tech.sticky_bounds


def test_nfe_counter():
    opt = optimiser(obj)
    assert opt.nfe == 0
    [opt.obj(i) for i in range(10)]
    assert opt.nfe == 10


def test_post_step():
    opt = optimiser(obj)

    def log(op):
        return op.name

    opt.post_step = log
    assert opt.post_step(opt) == "optimiser"


@pytest.mark.parametrize("s", [np.NaN, np.inf, None, [], True, False])
def test_bad_obj(s):
    def bad_obj(x):
        return s

    opt = optimiser(obj=bad_obj)
    assert opt.obj(None) == freelunch.base._BAD_OBJ_SCORE


@pytest.mark.parametrize("s", [np.int32(1), np.float64(1.2), int(0), float()])
def test_good_obj(s):
    def good_obj(x):
        return s

    opt = optimiser(obj=good_obj)
    assert opt.obj(None) == s
