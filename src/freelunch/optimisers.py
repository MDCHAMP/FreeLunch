"""
Main module definitions in here

"""
import numpy as np

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

        # Test Initial Population
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

        return sorted([ p.as_sol() for p in pop ], key=lambda p: p.fitness)

    


