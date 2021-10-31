'''
New home for the zoo of optimisation creatures
'''
import numpy as np

from freelunch.util import InvalidSolutionUpdate, BadObjectiveFunctionScores
from freelunch.util import verify_real_finite


# %% Base class

class animal:
    '''
    Handy dandy common object for storing trial solutions / other interesting data
    '''

    def __init__(self, dna=None, fitness=None, best=None, best_pos=None):
        
        
        self.best= best
        self.best_pos = best_pos
        self.dna = dna
        self._fitness = None
        self.fitness = fitness
        self.tech = []

    def move(self, dna, fitness):
        if np.all(np.isreal(dna)) and\
             not np.any(np.isinf(dna)) and\
             not np.any(np.isnan(dna)):
            self.dna = dna
            self.fitness = fitness
        else:
            raise InvalidSolutionUpdate

    @property
    def fitness(self):
        return self._fitness

    
    @fitness.setter
    @verify_real_finite([1], ['fitness'])
    def fitness(self, fitness):
        if (fitness is not None) and ((self._fitness is None) or (fitness < self.best)):
            self.best = fitness
            self.best_pos = self.dna
        self._fitness = fitness
    
    def on_win(self):
        '''big brain time'''
        for t in self.tech:
            if isinstance(t, tuple):
                t, v, = t
                t.win(v)
            else:
                t.win()
        self.tech = []

    def __lt__(self, other):
        '''overlaod the < operator for convenient handling of tournament selction in the presence onf nonetype fitnesses'''
        if self.fitness is None:
            if other.fitness is None:
                # ? what to do here - raise error TR
                # return False
                raise BadObjectiveFunctionScores
            return False # Other has lower fitness
        elif other.fitness is None:
            return True # We have the lower fitness            
        else:
            return self.fitness < other.fitness
        
    def __gt__(self, other):
        '''overlaod the > operator for convenient handling of tournament selction in the presence onf nonetype fitnesses'''
        if self.fitness is None:
            if other.fitness is None:
                # ? what to do here - raise error TR
                # return False
                raise BadObjectiveFunctionScores
            return True # Other has lower fitness
        elif other.fitness is None:
            return False # We have the lower fitness            
        else:
            return self.fitness > other.fitness

    def __eq__(self, o: object) -> bool:
        # Not the same class
        if type(self) != type(o):
            return False
        # Get user attributes from self and o
        us = {k: v for k, v in self.__dict__.items()}
        them = {k: v for k, v in o.__dict__.items()}

        for k in us.keys():
            if np.any(us[k] != them[k]):
                return False
        
        # If not false must be true
        return True
    
    def as_sol(self):
        return animal(dna=self.best_pos, fitness=self.best)


# %% All that the light touches is our domain
class particle(animal):
    '''
    Want to store info on particles in a swarm? I got you bud
    '''

    def __init__(self, pos=None, vel=None, fitness=None, best=None, best_pos=None):
        
        super().__init__(dna=pos, fitness=fitness, best=best, best_pos=best_pos)
        self.vel = vel

    
    def move(self, pos, vel, fitness):
        if np.all(np.isreal(pos)) and\
             not np.any(np.isinf(pos)) and\
             not np.any(np.isnan(pos)) and\
             np.all(np.isreal(vel)) and\
             not np.any(np.isinf(vel)) and\
             not np.any(np.isnan(vel)):
            self.pos = pos
            self.vel = vel
            self.fitness = fitness
        else:
            raise InvalidSolutionUpdate

    @property
    def dna(self):
        return self.pos
    
    @dna.setter
    def dna(self, dna):
        self.pos = dna       


class krill(particle):

    '''
    I am a krill, a type of animal 

    I am also basically just a particle...
    '''

    def __init__(self, pos=None, vel=None, fitness=None, best=None, best_pos=None, motion=None, forage=None):
        
        super().__init__(pos=pos, vel=vel, fitness=fitness, best=best, best_pos=best_pos)

        self.motion = motion
        self.forage = forage
