'''
New home for the zoo of optimisation creatures
'''



# %% Base class

class animal:
    '''
    Handy dandy common object for storing trial solutions / other interesting data
    '''

    def __init__(self, dna=None, fitness=None, best=None, best_pos=None):
        self.dna = dna
        self.fitness = fitness
        self.tech = []

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
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
                # ? what to do here
                return False
            return False # Other has lower fitness
        elif other.fitness is None:
            return True # We have the lower fitness            
        else:
            return self.fitness < other.fitness
        
# %% All that the light touches is our domain




class particle(animal):
    '''
    Want to store info on particles in a swarm? I got you bud
    '''

    def __init__(self, pos=None, vel=None, fitness=None, best=None, best_pos=None):
        self.pos = pos
        self.vel = vel
        self._fitness = None
        self.fitness = fitness
        self.best = best
        self.best_pos = best_pos

    @property
    def dna(self):
        return self.pos

    def as_sol(self):
        sol = animal()
        sol.dna = self.best_pos
        sol.fitness = self.best
        return sol


class krill(animal):

    '''
    I am a krill, a type of animal 

    I am also basically just a particle...
    '''

    def __init__(self, pos=None, vel=None, fitness=None, best=None, best_pos=None, motion=None, forage=None):
        self.pos = pos
        self.vel = vel
        self._fitness = None
        self.fitness = fitness
        self.best = best
        self.best_pos = best_pos
        self.motion = motion
        self.forage = forage

    @property
    def dna(self):
        return self.pos

    def as_sol(self):
        sol = animal()
        sol.dna = self.best_pos
        sol.fitness = self.best
        return sol
