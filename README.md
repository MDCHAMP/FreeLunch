# FreeLunch - Meta-heuristic optimisation suite for python


[![Build Status](https://travis-ci.com/MDCHAMP/FreeLunch.svg?branch=main)](https://travis-ci.com/MDCHAMP/FreeLunch) [![codecov](https://codecov.io/gh/MDCHAMP/FreeLunch/branch/main/graph/badge.svg)](https://codecov.io/gh/MDCHAMP/FreeLunch)

Basically a dump of useful / funny metaheurisitcs with a (hopefully) simpe interface

Feeling cute might add benchmarking later idk

There are literally so many implementations of all of these so... here's one more!

## Features

### Optimisers

Your favourite not in the list? Feel free to add it.

- Differential evolution `freelunch.DE`
- Simulated Annealing `freelunch.SA`

--Coming soon to 0.1.0--

- SADE
- PSA
- Quantum Bees
- Grenade Eplosion Method
- The Penguin one


### Benchmarking functions

Tier list: TBA

- N-dimensional Ackley function
- N-dimensional Periodic function
- N-dimensional Happy Cat function
- N-dimensional Exponential function

## Usage

### Optimisers

Install with pip (req. numpy).

```
pip install freelunch
```

Import and instance your favourite meta-hueristics!

```python
import freelunch
opt = freelunch.DE(obj=my_objective_function, bounds=my_bounds) # Differential evolution
```

`obj` - objective function, callable: `obj(sol) -> float or None`
`bounds` - bounds for elements of sol: `bounds [[lower, upper]]*len(sol)` where: `(sol[i] <= lower) -> bool` and `(sol[i] >= upper) -> bool`.

Check out the hyperparameters and set your own, (defualts set automatically):

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

Run by calling the instance. To return the best solution only:

```python
quick_result = opt() # Calls optimiser.run_quik() is it exists which can be faster
                     # This can be checked with class.can_run_quick = bool
```

To return optimum after `nruns`:

```python
best_of_runs = opt(nruns=n) 
```

Return best `m` solutions in `np.ndarray`:

```python
best_m = opt(return_m=m)
```

Return json friendly dict with fun metadata!

```python
full_output = opt(full_output=True)
    # {
    #     'optimiser':'DE',
    #     'hypers':...,
    #     'bounds':...,
    #     'nruns':nruns,
    #     'nfe':1234,
    #     'solutions':[sol1, sol2, ..., solm*nruns],
    #     'scores':[fit1, fit2, ..., fitm*nruns]
    # }

```

### Benchmarks 

Access from `freelunch.benchmakrs` for example:

```python
bench = freelunch.benchmarks.ackley(n=2) # 2D ackley function

fit = bench(sol) # evaluate by calling
fit.default_bounds(n) # [[-10, 10],[-10, 10]]
fit.optimum(n) # [0, 0]
fit.f0 # 0.0
```

