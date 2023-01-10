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
    tol=None

    def __init__(self, n=2):
        self.n = n
        self.bounds = self.default_bounds() 
        self.optimum = self.rtn_optimum()

    def __call__(self, dna):
        if len(dna) == 0:
            raise ZeroLengthSolutionError('An empty trial solution was passed')
        return self.obj(dna)

# %% some misc (v0.x) benchmarks

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



# %% https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf

class DeJong(benchmark):
    '''
    DeJong's 1st function in n dimensions
    '''
    default_bounds = lambda self:np.array([[-5.12, 5.12]]*self.n)
    rtn_optimum = lambda self:np.array([0]*self.n)
    f0 = 0

    def obj(self, dna):
        return np.sum(dna**2)

class HyperElipsoid(benchmark):
    '''
    HyperElipsoid function in n dimensions
    '''
    default_bounds = lambda self:np.array([[-5.12, 5.12]]*self.n)
    rtn_optimum = lambda self:np.array([0]*self.n)
    f0 = 0

    def obj(self, dna):
        return np.sum(np.arange(1,self.n+1)*dna**2)

class RotatedHyperElipsoid(benchmark):
    '''
    RotatedHyperElipsoid function in n dimensions
    '''
    default_bounds = lambda self:np.array([[-65.536, 65.536]]*self.n)
    rtn_optimum = lambda self:np.array([0]*self.n)
    f0 = 0

    def obj(self, dna):
        out = 0
        for i in range(self.n):
            out += np.sum(dna[:i+1]**2)
        return out

class Rosenbrock(benchmark):
    '''
    Rosenbrock's function in n dimensions (banana function)
    '''
    default_bounds = lambda self:np.array([[-2.048, 2.048]]*self.n)
    rtn_optimum = lambda self:np.array([1]*self.n)
    f0 = 0

    def obj(self, dna):
        return np.sum(100*(dna[1:] - dna[:-1]**2)**2 + (1-dna[:-1])**2)
        

class Ragstrin(benchmark):
    '''
    Ragstrin's function in n dimensions 
    '''
    default_bounds = lambda self:np.array([[-5.12, 5.12]]*self.n)
    rtn_optimum = lambda self:np.array([0]*self.n)
    f0 = 0
    
    def obj(self, dna):
        return 10 * self.n + np.sum(dna**2 - 10*np.cos(2* np.pi*dna))
        
class Schwefel(benchmark):
    '''
    Schwefel's function in n dimensions

    (divided through by dimension for constant f0)
    '''
    default_bounds = lambda self:np.array([[-500, 500]]*self.n)
    rtn_optimum = lambda self:np.array([420.9687]*self.n)
    f0 = 0
    
    def obj(self, dna):
        return 418.9828872721625-np.sum(dna*np.sin(np.sqrt(np.abs(dna))))/self.n
        
class Griewangk(benchmark):
    '''
    Griewangk's function in n dimensions

    '''
    default_bounds = lambda self:np.array([[-600, 600]]*self.n)
    rtn_optimum = lambda self:np.array([0]*self.n)
    f0 = 0
    
    def obj(self, dna):
        return (1/4000)* np.sum(dna**2) - np.prod(np.cos(dna/np.sqrt(np.arange(self.n)+1))) + 1 

class PowerSum(benchmark):
    '''
    Powersum function in n dimensions

    '''
    default_bounds = lambda self:np.array([[-1, 1]]*self.n)
    rtn_optimum = lambda self:np.array([0]*self.n)
    f0 = 0
    
    def obj(self, dna):
        out = 0
        for i,x in enumerate(dna):
            out+=np.abs(x)**(i+1) 
        return out

class Ackley(benchmark):
    '''
    Ackely function in n dimensions
    '''

    default_bounds = lambda self:np.array([[-32.768, 32.768]]*self.n)
    rtn_optimum = lambda self:np.array([0]*self.n)
    f0 = 0

    a,b,c = 20, 0.2, 2*np.pi
    def obj(self, dna):
        t1 = -self.a * np.exp(-self.b * (1/len(dna)) * np.sum(dna**2))
        t2 = - np.exp(1/len(dna) * np.sum(np.cos(self.c * dna))) 
        t3 = self.a + np.exp(1)
        return t1 + t2 + t3   


MOLGA_TEST_SUITE = [
    DeJong,
    HyperElipsoid,
    RotatedHyperElipsoid,
    Rosenbrock,
    Ragstrin,
    Schwefel,
    Griewangk,
    PowerSum,
    Ackley
]