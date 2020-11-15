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

### Benchmarking functions

Tier list: TBA

- N-dimensional Ackley function
- N-dimensional Periodic function
- N-dimensional Happy Cat function
- N-dimensional Exponential function

## Usage

Install with pip (req. numpy).

```
pip install freelunch
```

Import and instance your favourite meta-hueristics!

```python
import freelunch
opt = freelunch.DE(obj=my_objective_function, bounds=my_bounds)
```

Return best solution only.

```python
quick_result = opt()
```

Return optimum after `nruns`

```python
best_of_runs = opt(nruns = n) 
```

Return best `m` solutions in `np.ndarray`
```python
best_m = opt(return_m = m)
```

Return json friendly dict with fun metadata!

```python
full_output = opt(full_output = True)
```
