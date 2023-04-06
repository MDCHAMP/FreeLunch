"""
Main module definitions in here
"""
import warnings

import numpy as np

from freelunch import tech
from freelunch.base import optimiser



# %% Optimisation classes

class PSO(optimiser):
    '''
    Base Class for Particle Swarm Optimisations
    '''
    name = 'Particle Swarm Optimisation'
    tags = ['Continuous domain', 'Particle swarm']
    hyper_definitions = {
        'N':'Population size (int)',
        'G':'Number of generations (int)',
        'I':'Inertia coefficients (np.array, I.shape=(2,))',
        'A':'Acceleration (np.array, I.shape=(2,))',
        'v':'Velocity bounds'
    }
    hyper_defaults = {
        'N':100,
        'G':100,
        'I':np.array([0.1, 0.9]),
        'A':np.array([0.1, 0.1]),
        'v':np.array([-1, 1])
    }

    # required functions

    def pre_loop(self):
        self.pos = tech.uniform_continuous_init(self.bounds, self.hypers['N'])
        self.vel = tech.uniform_continuous_init([self.hypers['v']]*len(self.bounds), self.hypers['N'])
        self.fit = np.array([self.obj(x) for x in self.pop])
        idx = np.argmin(self.fit)
        self.local_best_pos = self.pos.copy()
        self.global_best_pos = self.pos[idx].copy()
        self.local_best_fit = self.fit.copy()
        self.global_best_fit = self.fit[idx]

    def step(self):
        # Basic particle swarm move
        inertia = tech.lin_reduce(self.hypers['I'], self.gen, self.hypers['G'])
        # inertia = self.hypers['I'][1]-(self.hypers['I'][1]-self.hypers['I'][0])*self.gen/self.hypers['G']
        # Update velocity
        self.vel = (
            inertia*self.vel + 
            self.hypers['A'][0]*np.random.rand((self.hypers["N"],))*(self.local_best_pos - self.pos) + 
            self.hypers['A'][1]*np.random.rand((self.hypers["N"],))*(self.global_best_pos - self.pos))
        # Move
        self.pos += self.vel

        # Bounding
        self.bounder(self)

        # Eval fitness
        self.fit = np.array([self.obj(x) for x in self.pop])
        
        # Bookeeping post eval
        tech.update_local_best(self)
        tech.update_global_best(self)


class QPSO(PSO):
    '''
    Quantum Particle Swarm Optimisations
    '''
    name = 'Quantum Particle Swarm Optimisation'
    tags = ['Continuous domain', 'Particle swarm']
    hyper_definitions = {
        'N': 'Population size (int)',
        'G': 'Number of generations (int)',
        'alpha': 'Contraction Expansion Coefficient (np.array, I.shape=(2,))',
        'v':'Velocity bounds',
    }
    hyper_defaults = {
        'N': 100,
        'G': 100,
        'alpha': np.array([1.0, 0.5]),
        'v':np.array([-1, 1])
    }

    def step(self, pop, gen):
        C = self.local_best_pos.mean(0)
        D = len(C)
        alpha = tech.lin_reduce(self.hypers['alpha'], self.gen, self.hypers['G'])

        phi = np.random.uniform(size=(self.hypers["N"],))
        u = np.random.uniform(size=(self.hypers["N"],))
        self.pos = (
            phi*self.local_best_pos + (1-phi)*self.global_best_pos + 
            np.sign(np.random.normal(size=(self.hypers["N"],D))) * 
            alpha*np.abs(C-self.pos)*np.log(1/u)
            )
        
        # Bounding
        self.bounder(self)

        # Eval fitness
        self.fit = np.array([self.obj(x) for x in self.pop])
        
        # Bookeeping post eval
        tech.update_local_best(self)
        tech.update_global_best(self)
    

class ABC(optimiser):
    '''
    Artificial Bee Colony
    '''
    name = 'Artificial Bee Colony Optimisation'
    tags = ['Continuous domain', 'Particle swarm', 'Utterly terrible']
    hyper_definitions = {
        'N': 'Population size (int)',
        'G': 'Number of generations (int)',
    }
    hyper_defaults = {
        'N': 100,
        'G': 100,
    }

    def pre_loop(self):
        self.pos = tech.uniform_continuous_init(self.bounds, self.hypers['N'])
        self.fit = np.array([self.obj(x) for x in self.pop])
        
    def step(self):

        # New candidates
        crossover_point = self.pos.copy()
        N = self.hypers['N']
        
        # Employed Bees Phase
        d = np.random.randint(0,self.pos.shape[1],size=self.pos.size[0])
        rand_idx = np.arange(N) + np.random.randint(1,N,size=(N))
        rand_idx[rand_idx>=N] -= N
        crossover_point[rand_idx,d] = self.pos[rand_idx,d]
        new_candidates = self.pos + np.random.uniform(-1,1,size=(self.hypers["N"]))*crossover_point

        # Greedy Selection...
        new_fit = np.array([self.obj(x) for x in new_candidates])
        better_idx = new_fit < self.fitness
        fitness = self.fitness.copy()
        fitness[better_idx] = new_fit[better_idx]
        pos = self.pos.copy()
        pos[better_idx] = new_candidates[better_idx]

        # Eq 7...
        fit_xm = np.ones_like(fitness)
        fit_xm[fitness >= 0] = 1/(1+fitness[fitness >= 0])
        fit_xm[fitness < 0] = 1+np.abs(fitness[fitness<0])

        # Enter the onlooker bee phase
        onlookers = np.random.choice(
            pos,
            size=N,
            p=fit_xm/np.sum(fit_xm),
            replace=True)

        crossover_point = onlookers.copy()
        d = np.random.randint(0,self.pos.shape[1],size=self.pos.size[0])
        rand_idx = np.arange(N) + np.random.randint(1,N,size=(N))
        rand_idx[rand_idx>=N] -= N
        crossover_point[rand_idx,d] = self.pos[rand_idx,d]

        new_candidates = onlookers + np.random.uniform(-1,1,size=(self.hypers["N"]))*crossover_point

        # Greedy Selection...
        new_fit = np.array([self.obj(x) for x in new_candidates])
        better_idx = new_fit < self.fitness
        fitness = self.fitness.copy()
        fitness[better_idx] = new_fit[better_idx]
        pos = self.pos.copy()
        pos[better_idx] = new_candidates[better_idx]

        


