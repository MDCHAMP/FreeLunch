'''
Benchmarks for testing / comparisons
'''

import numpy as np
from freelunch.util import ZeroLengthSolutionError

# %%

class benchmark:
    '''
    base class for benchmarking functions
    '''
    default_bounds = lambda n:None
    rtm_optimum = lambda n:None

    def __init__(self, n=None):
        if n is None:
            self.n = 2
        else:
            self.n = n
        self.bounds = self.default_bounds() 
        self.optimum = self.rtn_optimum()

    def __call__(self, dna):
        if len(dna) == 0:
            raise ZeroLengthSolutionError('An empty trial solution was passed')
        return self.obj(dna)

# %%


class ackley(benchmark):
    '''
    ackely function in n dimensions
    '''

    default_bounds = lambda self:np.array([[-10, 10]]*self.n)
    rtn_optimum = lambda self:np.array([0]*self.n)
    f0 = 0
    tol = 10**-2

    a,b,c = 20, 0.2, 2*np.pi
    def obj(self, dna):
        t1 = -self.a * np.exp(-self.b * (1/len(dna)) * np.sum(dna**2))
        t2 = - np.exp(1/len(dna) * np.sum(np.cos(self.c * dna))) 
        t3 = self.a + np.exp(1)
        return t1 + t2 + t3


class exponential(benchmark):
    '''
    exponential function in n dimensions
    '''
    
    default_bounds = lambda self:np.array([[-10, 10]]*self.n)
    rtn_optimum = lambda self:np.array([0]*self.n)
    f0 = -1
    tol = 10**-3

    a = -0.5
    def obj(self, dna):
        t1 = - np.exp(self.a * np.sum(dna**2))
        return t1


class happycat(benchmark):
    '''
    happycat function in n dimensions
    '''
    
    default_bounds = lambda self:np.array([[-2, 2]]*self.n)
    rtn_optimum = lambda self:np.array([-1]*self.n)
    f0 = 0
    tol = None

    a = 1/8
    def obj(self, dna):
        norm = np.sum(dna**2)
        t1 = ((norm - self.n)**2)**(self.a)
        t2 = (1/self.n)*(0.5*norm + np.sum(dna)) + 0.5
        return t1 + t2


class periodic(benchmark):
    '''
    periodic function in n dimensions
    '''
    
    default_bounds = lambda self:np.array([[-10, 10]]*self.n)
    rtn_optimum = lambda self:np.array([0]*self.n)
    f0 = 0.9
    tol = None

    def obj(self, dna):
        t1 = np.sum(np.sin(dna)**2) +1
        t2 = - 0.1 * np.exp(-np.sum(dna**2))
        return t1 + t2

