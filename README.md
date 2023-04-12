# FreeLunch - Meta-heuristic optimisation suite for python


[![PyPI](https://badge.fury.io/py/freelunch.svg)](https://badge.fury.io/py/freelunch)![Code](https://github.com/MDCHAMP/FreeLunch/workflows/actions%20code%20quality/badge.svg) ![Tests](https://github.com/MDCHAMP/FreeLunch/workflows/actions%20pytest/badge.svg)   ![Benchmark](https://github.com/MDCHAMP/FreeLunch/workflows/actions%20pytest%20benchmark/badge.svg) [![Coverage](https://codecov.io/gh/MDCHAMP/FreeLunch/branch/main/graph/badge.svg)](https://codecov.io/gh/MDCHAMP/FreeLunch)

*Please note the minor changes to the `optimiser` call signature since `0.0.11`, details below.* 
___
## About

Freelunch is a convenient python implementation of a number of meta-heuristic optimisation (with an 's') algorithms.  

___

## Features

### Optimisers

--NEW and EXCITING---
 - Particle attractor optimistion `freelunch.PAO` (pronounced POW)

Your favourite not in the list? Feel free to add it.

- Differential evolution `freelunch.DE`
- Simulated Annealing `freelunch.SA`
- Particle Swarm `freelunch.PSO`
- Krill Herd `freelunch.KrillHerd`
- Self-adapting Differential Evolution `freelunch.SADE`
- Quantum Particle Swarm `freelunch.QPSO` 

--Coming soon to 0.1.0--

- Quantum Bees
- Grenade Explosion Method
- The Penguin one


### Benchmarking functions

Tier list: TBA

- N-dimensional Ackley function
- N-dimensional Periodic function
- N-dimensional Happy Cat function
- N-dimensional Exponential function


___
## Install

Install with pip (req. `numpy`).

```
pip install freelunch
```
___
## Usage

Create instances of your favourite meta-heuristics!

```python
import freelunch
opt = freelunch.DE(my_objective_function, bounds=my_bounds) # Differential evolution
```

Where,

 - `obj`: objective function that excepts a single argument, the trial vector `x`, and returns `float ` or `None`. i.e. `obj(x) -> float or None`


 - `bounds`: Iterable bounds for elements of `x` i.e. `bounds [[lower, upper]]*len(sol)` 
where: `(sol[i] <= lower) -> bool` and `(sol[i] >= upper) -> bool`.


## Running the optimisation

Run by calling the instance. There are several different calling signatures. Use any combination of the arguments below to suit your needs! 


To return the best solution only:

```python
quick_result = opt() # (D,)
```

To return optimum after `n_runs`:

```python
best_of_nruns = opt(n_runs=n) # (D,)
```

To return optimum after `n_runs` in parallel (uses `multiprocessing.Pool`), see note below.:

```python
best_of_nruns = opt(n_runs=n, n_workers=w, pool_args={}, chunks=1) # (D,)
```

Return best `m` solutions in `np.ndarray`:

```python
best_m = opt(n_return=m) # (D, m)
```

Return `json` friendly `dict` with fun metadata!

```python
full_output = opt(full_output=True)
    # {
    #     'optimiser':'DE',
    #     'hypers':...,
    #     'bounds':...,
    #     'nruns':nruns,
    #     'nfe':1234,
    #     'solutions':[sol1, sol2, ..., solm*n_runs], # All solutions from all runs sorted by fitness
    #     'scores':[fit1, fit2, ..., fitm*n_runs]
    # }

```
___
## Customisation

Want to change things around?

- ### Change the initialisation strategy

Custom initialisation strategies can be supplied by altering the `optimiser.initialiser` attribute of any optimiser instance. For example:

```python
opt = fr.DE(obj, ...)

def my_initialiser(bounds, N, **hypers):
    'Custom population initialisation'
    # Remember to return and interable of length N
    population = ... # custum logic
    return population

```

Additionally, some examples of common initialisation strategies can be found in the `freelunch.tech` module. 

- ### Change the bounding strategy

The simplest way to do this is to overwrite the `optimiser.bounder` attribute. There are a number of ready made strategies in `freelunch.tech` or alternatively define a custom method with the following call signature. 

```python

opt = fr.DE(obj, ...)

def my_bounder(p, bounds, **hypers):
    '''custom bounding method'''
    p.dna = ... # custom bounding logic

opt.bounder = my_bounder # overwrite the bounder attribute

# and then call as before
x_optimised = opt()
```

 - ### Changing the hyperparameters

Check out the hyperparameters and set your own, (defaults set automatically):


```python
print(opt.hyper_definitions)
    # {
    #     'N':'Population size (int)',
    #     'G':'Number of generations (int)',
    #     'F':'Mutation parameter (float in [0,1])',
    #     'Cr':'Crossover probability (float in [0,1])'
    # }

print(opt.hyper_defaults)
    # {
    #     'N':100,
    #     'G':100,
    #     'F':0.5,
    #     'Cr':0.2
    # }

opt.hypers.update({'N':300})
print(opt.hypers)
    # {
    #     'N':300,
    #     'G':100,
    #     'F':0.5,
    #     'Cr':0.2
    # }
```
___
## Benchmarks 

Access from `freelunch.benchmarks` for example:

```python
bench = freelunch.benchmarks.ackley(n=2) # Instanciate a 2D ackley benchmark function

fit = bench(sol) # evaluate by calling
bench.bounds # [[-10, 10],[-10, 10]]
bench.optimum # [0, 0] 
bench.f0 # 0.0
```
___
## A note on running optimisations in parallel. 

Because `multiprocessing.Pool` relies on `multiprocessing.forking.pickle` to send code to parallel processes, it is imperative that anything passed to the freelunch optimisers can be pickled. For example, the following common python pattern for producing an objective function with a single argument,

```python

method = ... # some methods / args that are requred by the objective function
args = 

def wrap_my_obj(method, args):
    def _obj(x):
        return method(args, x)
    return _obj

obj = wrap_my_obj(method, args)

```

cannot be pickled because `_obj` is not importable from the top level module scope and will raise `freelunch.util.UnpicklableObjectiveFunction` . Instead consider using `functools.partial` i.e.


```python

from functools import partial

method = ... # some methods / args that are requred by the objective function
args = ...


def _obj(method, args, x):
    return method(args, x)

obj = partial(_obj, method, args)

```

