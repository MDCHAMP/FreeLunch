'''
Testing tech
'''
import pytest
import numpy as np

from freelunch.base import *
from freelunch.darwin import rand_1

def test_hyp_parse():

    opt = optimiser()
    assert(rand_1.__name__ == opt.parse_hyper(rand_1).__class__.__name__)
