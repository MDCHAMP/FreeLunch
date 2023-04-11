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
        self.vel = tech.uniform_continuous_init(np.array([self.hypers['v']]*len(self.bounds)), self.hypers['N'])
        self.fit = np.array([self.obj(x) for x in self.pos])
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
            self.hypers['A'][0]*np.random.uniform((self.hypers["N"],))*(self.local_best_pos - self.pos) + 
            self.hypers['A'][1]*np.random.uniform((self.hypers["N"],))*(self.global_best_pos - self.pos))
        # Move
        self.pos += self.vel

        # Bounding
        self.bounder(self)

        # Eval fitness
        self.fit = np.array([self.obj(x) for x in self.pos])
        
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

    def step(self):
        C = self.local_best_pos.mean(0)
        D = len(C)
        alpha = tech.lin_reduce(self.hypers['alpha'], self.gen, self.hypers['G'])

        phi = np.random.uniform(size=(self.hypers["N"],1))
        u = np.random.uniform(size=(self.hypers["N"],1))
        self.pos = (
            phi*self.local_best_pos + (1-phi)*self.global_best_pos + 
            np.sign(np.random.normal(size=(self.hypers["N"],D))) * 
            alpha*np.abs(C-self.pos)*np.log(1/u))
        
        # Bounding
        self.bounder(self)

        # Eval fitness
        self.fit = np.array([self.obj(x) for x in self.pos])
        
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
        'G': 'Number of generations (int)', # TODO add N_employed, N_onlookers etc. here also logic for making sure these sum to N
    }
    hyper_defaults = {
        'N': 100,
        'G': 100,
        'limit': 5,
    }

    def pre_loop(self):
        self.pos = tech.uniform_continuous_init(self.bounds, self.hypers['N'])
        self.fit = np.array([self.obj(x) for x in self.pos])
        self.best_fit = np.min(self.fit)
        self.best_pos = self.pos[np.argmin(self.fit)]
        self.stalls = np.zeros((self.hypers["N"]))
        self.hypers["N_employed"] = self.hypers["N"]//2
        self.hypers["N_onlookers"] = self.hypers["N"] - self.hypers["N_employed"]

    @staticmethod
    def crossover_points(pos):
        """ABC Crossover Location 

        Args:
            pos (np.ndarray): postitions to update

        Returns:
            np.ndarray: crossover points (x_{ik} - x_{jk})  
        """
        
        N = pos.shape[0]
        crossover_point = pos.copy()
        d = np.random.randint(0, pos.shape[1], size=(N,1))
        rand_idx = np.arange(N) + np.random.randint(1, N, size=(N,))
        rand_idx[rand_idx>=N] -= N
        mask = np.tile(np.arange(pos.shape[1]),(N,1)) == d
        crossover_point[mask] -= pos[rand_idx][mask]
        
        return crossover_point
        
    def step(self):

        # The employed bees (good capitalist bees)
        employed = self.pos[:self.hypers["N_employed"]]
        employed_fit = self.fit[:self.hypers["N_employed"]]
        crossover_point = self.crossover_points(employed) 
        new_candidates = employed + np.random.uniform(-1,1,size=(self.hypers["N_employed"],1))*crossover_point

        # Greedy Selection...
        new_fit = np.array([self.obj(x) for x in new_candidates])
        tech.greedy_selection(employed_fit, new_fit, employed, new_candidates)

        # "Fitness" of each employed bee 
        fit_xm = np.ones_like(employed_fit)
        fit_xm[employed_fit >= 0] = 1/(1+employed_fit[employed_fit >= 0])
        fit_xm[employed_fit < 0] = 1+np.abs(employed_fit[employed_fit < 0])

        # Enter the onlooker bee phase (bad unproductive bees)

        # Pinwheel onlookers choosing one of the employees to take credit for...
        idx = np.random.choice(
            np.arange(self.hypers["N_employed"]),
            size=self.hypers["N_onlookers"],
            p=fit_xm/np.sum(fit_xm),
            replace=True)
        onlookers = employed[idx]
        onlookers_fit = employed_fit[idx]

        # Onlooker selections
        crossover_point = self.crossover_points(onlookers) 
        new_candidates = onlookers + np.random.uniform(-1,1,size=(self.hypers["N_onlookers"],1))*crossover_point

        # Greedy Selection...
        new_fit = np.array([self.obj(x) for x in new_candidates])
        tech.greedy_selection(onlookers_fit, new_fit, onlookers, new_candidates)

        # Cleaning up
        new_pop = np.concatenate((employed, onlookers),axis=0)
        new_fit = np.concatenate((employed_fit, onlookers_fit),axis=0)
        self.stalls[np.all(self.pos == new_pop, axis=1)] += 1
        self.pos = new_pop
        self.fit = new_fit
        
        if np.any(self.fit < self.best_fit):
            self.best_fit = np.min(self.best_fit)
            self.best_pos = self.pos[np.argmin(self.fit)]




       

        


