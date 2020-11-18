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



