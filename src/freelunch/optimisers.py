"""
Main module definitions in here

"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
from abc import ABC, abstractmethod

from freelunch import tech
from freelunch.base import continuous_space_optimiser

class DE(continuous_space_optimiser):
    '''
    Differential evolution
    '''
    name = 'Differential Evolution'
    tags = ['continuous domain', 'population based', 'evolutionary']
    hyper_definitions = {
        'N':'Population size (int)',
        'G':'Number of generations (int)',
        'F':'Mutation parameter (float in [0,1])',
        'Cr':'Crossover probability (float in [0,1])'
    }
    hyper_defaults = {
        'N':100,
        'G':100,
        'F':0.5,
        'Cr':0.2
    }

    def run(self):
        pop = tech.uniform_continuous_init(self.bounds, self.hypers['N'])
        tech.compute_obj(pop, self.obj)
        for gen in range(self.hypers['G']):
            trial_pop = np.empty_like(pop, dtype=object)
            for i, sol in enumerate(pop):
                pts = np.random.choice(pop, 3)
                trial = tech.solution()
                trial.dna = (pts[0].dna - pts[1].dna) + self.hypers['F'] * pts[2].dna
                trial.dna = tech.binary_crossover(sol.dna, trial.dna, self.hypers['Cr'])
                trial.dna = tech.apply_sticky_bounds(trial.dna, self.bounds)
                trial_pop[i] = trial
            tech.compute_obj(trial_pop, self.obj)
            pop = tech.sotf(pop, trial_pop)
            jrand = np.random.randint(0, self.hypers['N'])
            pop[jrand] = trial_pop[jrand]
        return pop


class SA(continuous_space_optimiser):
    '''
    Simulated Annealing 
    '''
    name = 'Simulated Annealing'
    tags = ['Continuous domain', 'Annealing']
    hyper_definitions = {
        'K':'number of timesteps (int)',
        'T':'Temperatures (np.ndarray, T.shape=(k,))',
        'P':'Acceptance probability (P(e, e, T) -> bool)'
    }

    def run(self):
        raise NotImplementedError


class PSO(continuous_space_optimiser):
    '''
    Base Class for Particle Swarm Optimisations
    '''

    name = 'Particle Swarm Optimisation'
    tags = ['Continuous domain', 'Particle swarm']
    hyper_definitions = {
        'N':'Population size (int)',
        'G':'Number of generations (int)',
        'I':'Inertia coefficients (np.array, I.shape=(2,))',
        'A':'Acceleration (np.array, I.shape=(2,))'
    }
    hyper_defaults = {
        'N':100,
        'G':200,
        'I':np.array([0.1, 0.9]),
        'A':np.array([0.1, 0.1])
    }

    def init_pop(self,N):
        # Function which initialises the swarm
        pop = np.empty((N,), dtype=object)
        for i in range(N):
            pop[i] = tech.particle(np.array([np.random.uniform(a,b) for a, b in self.bounds]))
            pop[i].vel = np.squeeze((2*np.random.rand(self.bounds.shape[0],1)-1)*np.diff(self.bounds))
        return pop

    def move_swarm(self, pop, gen):
        # Basic particle swarm move

        inertia = self.hypers['I'][1]-(self.hypers['I'][1]-self.hypers['I'][0])*gen/self.hypers['G']
        # This loop is slow, should vectorise at some point
        for p in pop:
            p.vel = inertia*p.vel + \
                self.hypers['A'][0]*np.random.rand()*(p.best_pos-p.pos) + \
                self.hypers['A'][1]*np.random.rand()*(self.g_best.pos-p.pos)
            p.pos = tech.apply_sticky_bounds(p.pos + p.vel, self.bounds)
        return pop

    def test_pop(self, pop):
        # Test all population and update bests
        tech.compute_obj(pop, self.obj)
        return pop

    def best_particle(self, pop):
        # Particles particles on the wall, who's the bestest of them all
        best = tech.particle()
        loc = np.squeeze(np.where(pop==min(pop,key=lambda x: x.best)))
        best.fitness = pop[loc].best
        best.pos = pop[loc].best_pos
        return best

    # Generic PSO run routine
    def run(self):

        # Initialise the swarm
        pop = self.init_pop(self.hypers['N'])

        # Test Inital Population
        self.test_pop(pop)
        self.g_best = self.best_particle(pop)

        # Main loop
        for gen in range(self.hypers['G']):

            # Propagate the swarm
            pop = self.move_swarm(pop,gen)

            # Test new swarm locations
            self.test_pop(pop)

            # Particle class updates best previous position
            # Update global best
            self.g_best = self.best_particle(pop)

        return [ p.as_sol() for p in pop ]

    
class KrillHerd(continuous_space_optimiser):
    '''
    Krill Herd Optimisation

    Krill move based on three things:
        1) Induced motion
        2) Foraging
        3) Random Physical Diffusion
    
    dX/dt = N + F + D

    Gandomi, Amir Hossein, and Amir Hossein Alavi. "Krill herd: a new bio-inspired optimization algorithm." Communications in nonlinear science and numerical simulation 17.12 (2012): 4831-4845.

    '''

    name = 'Krill Herd'
    tags = ['Continuous domain', 'Animal', 'Krill Herd']
    hyper_definitions = {
        'N':'Population size (int)',
        'G':'Number of generations (int)',
        'I':'Inertia coefficients (np.array, I.shape=(2,))',
        'A':'Acceleration (np.array, I.shape=(2,))'
    }
    hyper_defaults = {
        'N':100,
        'G':200,
        'Imotion':np.array([0.9, 0.1]), 
        'Iforage':np.array([0.9, 0.1]), 
        'eps':1e-12,
        'Nmax':0.01 # confusingly this is maximum induced speed in the paper
    }

    def init_pop(self,N):
        # Function which initialises the krill randomly within the bounds 
        self.pop = np.empty((N,), dtype=object)
        for i in range(N):
            self.pop[i] = tech.krill( \
                pos = np.array([np.random.uniform(a,b) for a, b in self.bounds]) \
                motion = np.array([np.random.uniform(a,b) for a, b in self.bounds]) \
                forage = np.array([np.random.uniform(a,b) for a, b in self.bounds]) \
                )

        # Compute first set of fitness
        tech.compute_obj(self.pop, self.obj)

    def get_herd(self,pop):
        '''
        It is more convenient to work with matrix representations of the krill
        '''

        vals = np.zeros(self.hypers['N'])
        locs = np.zeros((self.hypers['N'],pop[0].pos.shape[0]))

        for i,krill in enumerate(pop):
            vals[i] = krill.fitness
            locs[i,:] = krill.pos
            motion[i,:] = krill.motion
            forage[i,:] = krill.forage

        return (vals,locs,motion,forage)

    def get_pop(self,herd):
        '''
        But the good times have to end and we go back to the population
        '''
        for i,krill in enumerate(herd):
            self.pop[i].fitness = krill[0]
            self.pop[i].pos = krill[1]
            self.pop[i].motion = krill[2]
            self.pop[i].forage = krill[3]

    def winners_and_losers(self,herd):
        '''
        Sometimes in life you're the best sometimes you're the worst, sorry krill
        '''
        win_idx = np.argmin(herd[0])
        lose_idx = np.argmax(herd[0])
        
        # Winner and loser are tuples of best/worst (fitness,location)
        winner = (herd[0][win_idx], herd[1][win_idx,:])
        loser = (herd[0][lose_idx], herd[1][lose_idx,:])

        return (winner,loser)

    def local_motion(self,herd,gen):

        # pairwise distances between krill
        dists = squareform(pdist(herd[1]))

        # Who's my neighbour
        sense_dist = np.sum(dists,axis=1)/5/self.hypers['N']
        neighbours = dists <= sense_dist

        winner, loser = self.winners_and_losers(herd)
        spread = loser[0] - winner[0]

        # Alpha stores local [0] and target [1] for each krill 
        alpha = [np.zeros_like(herd[1]), np.zeros_like(herd[1])]

        # Alpha local, the effect of the neighbours
        for i in range(dists.shape[0]):
            Khat = (herd[0][i] - herd[0][neighbours[i,:]]) / spread
            Xhat = (herd[1][neighbours[i,:],:] - herd[1][i,:]) / (dists[i,neighbours[i,:]] + self.hypers['eps'])
            alpha[0][i,:] = np.sum( Xhat * Khat )

        # Alpha target, take me to your leader
        Kbest = (herd[0] - winner[0]) / spread
        Xbest = (winner[1] - herd[1]) / (dists[:,np.argmin(herd[0])][:,None] + self.hypers['eps'])
        alpha[1] = Kbest[:,None]*Xbest

        # Exploration/exploitation coefficient
        Cbest = 2*(np.random.rand() + gen / self.hypers['G'])

        # Alpha is weighted combination of local and target
        alpha = alpha[0] + Cbest*alpha[1]

        inertia  = tech.lin_reduce(self.hypers['Imotion'],gen,self.hypers['G']) 
        return self.hypers['Nmax']*alpha + inertia*herd[2]
        

    def run(self):

        self.init_pop(self.hypers['N'])

        herd = self.get_herd(self.pop)

        # Main loop 
        for gen in range(self.hypers['G']):
            self.local_motion(herd,gen)

