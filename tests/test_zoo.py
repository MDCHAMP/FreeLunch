'''
Testing of the zoo
'''
import pytest
import numpy as np

np.random.seed(100)

from freelunch.zoo import animal, particle, krill
from freelunch.util import BadObjectiveFunctionScores, InvalidSolutionUpdate

animals = [particle, krill]

def test_animal():

    location_1 = np.array([1,1,1])
    fitness_1 = 2
    location_2 = np.array([0,0,0])
    fitness_2 = 0
    location_3 = np.array([2,2,2])
    fitness_3 = 10

    friend = animal(dna=location_1, fitness=fitness_1)
    assert(np.all(friend.dna == location_1))
    assert(friend.fitness == 2)
    assert(np.all(friend.best_pos == location_1))
    assert(friend.best == 2)

    friend.move(location_2, fitness_2)
    assert(np.all(friend.dna == location_2))
    assert(friend.fitness == 0)
    assert(np.all(friend.best_pos == location_2))
    assert(friend.best == 0)

    friend.move(location_3, fitness_3)
    assert(np.all(friend.dna == location_3))
    assert(friend.fitness == 10)
    assert(np.all(friend.best_pos == location_2))
    assert(friend.best == 0)

    with pytest.raises(ValueError):
        friend.move(location_3,None)
    
    with pytest.raises(ValueError):    
        friend.move(location_3, np.inf)

    with pytest.raises(ValueError):
        friend.move(location_3, np.nan)

    with pytest.raises(ValueError):
        friend.move(location_3, [])

    with pytest.raises(InvalidSolutionUpdate):
        friend.move(np.array([np.inf,1,1]), 1)

    with pytest.raises(InvalidSolutionUpdate):
        friend.move(np.array([np.nan,1,1]), 1)
    
    with pytest.raises(InvalidSolutionUpdate):
        friend.move(np.array([1+2j,1,1]), 1)

    friend = animal(dna=location_1, fitness=fitness_1)
    friend2 = animal(dna=location_2, fitness=fitness_2)
    assert(friend2 < friend)
    assert(friend > friend2)
    friend2._fitness = None # Or will throw error
    assert(friend < friend2)
    assert(not (friend2 < friend))
    assert(friend2 > friend)
    assert(not (friend > friend2))
    friend._fitness = None # Or will throw error

    with pytest.raises(BadObjectiveFunctionScores):
        friend < friend2
    
    with pytest.raises(BadObjectiveFunctionScores):
        friend > friend2


@pytest.mark.parametrize('creature', animals)
def test_particle(creature):

    location_1 = np.array([1,1,1])
    vel = np.random.randn(1,3)
    fitness_1 = 2
    location_2 = np.array([0,0,0])
    fitness_2 = 0
    location_3 = np.array([2,2,2])
    fitness_3 = 10

    friend = creature(pos=location_1, vel=vel, fitness=fitness_1)
    assert(np.all(friend.dna == location_1))
    assert(friend.fitness == 2)
    assert(np.all(friend.best_pos == location_1))
    assert(friend.best == 2)

    friend.move(location_2, vel, fitness_2)
    assert(np.all(friend.dna == location_2))
    assert(friend.fitness == 0)
    assert(np.all(friend.best_pos == location_2))
    assert(friend.best == 0)

    friend.move(location_3, vel, fitness_3)
    assert(np.all(friend.dna == location_3))
    assert(friend.fitness == 10)
    assert(np.all(friend.best_pos == location_2))
    assert(friend.best == 0)

    with pytest.raises(ValueError):
        friend.move(location_3,vel, None)
    
    with pytest.raises(ValueError):    
        friend.move(location_3, vel, np.inf)

    with pytest.raises(ValueError):
        friend.move(location_3, vel, np.nan)

    with pytest.raises(ValueError):
        friend.move(location_3,vel, [])

    with pytest.raises(InvalidSolutionUpdate):
        friend.move(np.array([np.inf,1,1]), vel, 1)

    with pytest.raises(InvalidSolutionUpdate):
        friend.move(np.array([np.nan,1,1]), vel, 1)
    
    with pytest.raises(InvalidSolutionUpdate):
        friend.move(np.array([1+2j,1,1]), vel, 1)

    with pytest.raises(InvalidSolutionUpdate):
        friend.move(location_1, np.array([np.inf,1,1]), 1)

    with pytest.raises(InvalidSolutionUpdate):
        friend.move(location_1, np.array([np.nan,1,1]), 1)
    
    with pytest.raises(InvalidSolutionUpdate):
        friend.move(location_1, np.array([1+2j,1,1]), 1)

    sol = friend.as_sol()
    assert(np.all(sol.dna == friend.best_pos))
    assert(sol.fitness == friend.best)

