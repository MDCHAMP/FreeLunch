# Welcome to freelunch!

We are always happy to work with anyone with a penchant for not paying for their second meal of the day.  

If you would like to contribute to freelunch please consider looking through the open issues. These are regularly updated with bugfixes and new algorithms to implement. 

If there is a specific algorithm/benchmark that you have in mind then please fork the repo and raise a PR. 

#### Design philosophy
- [Unga bunga python go brrrrr](https://xkcd.com/1513/)
- All inputs / outputs should be `JSON` friendly (for serialisation) .
- `optimiser.run` should be implemented with a similar abstraction level to the algorithm table in the original paper 
- Default hyperparameters should guarantee minimal benchmark performance.
- Common genetic operations (`mutation`, `crossover`, `selection`) should have a high level API and are 'hot-swappable'.
- Techniques used by >2 optimisers should be put into `freelunch.tech`.
- Class structure changes should only serve to make the above more straightforward.


#### Some guidelines for contributors
- Please derive new optimisers from a base class, try to be as descriptive as possible when doing this.
- Please implement the optimiser.run method in a readable style (See philosophy)
- Please make use of / add to the zoo!
- Sarcasm in comments is compulsory.


#### The module structure

`freelunch.base`

- Base optimiser class (optimiser) This class implements all of the basic API functionality for all derived optimiser classes, including handling hyperparameters and calls to the optimiser

- High level optimiser subclasses (continuous_space_optimiser, discrete_space_optimiser) Differentiate (semantically) the different types of optimiser.


`freelunch.optimisers`

- Implementations of the various optimisers provided in freelunch.


`freelunch.benchmarks`

- Implementations of the benchmarking functions, derived from base `benchmark` class. 


`freelunch.darwin`

- API for common genetic operations (`mutation`, `crossover`, `selection`).


`freelunch.zoo`

- Solution base class and classes for bio-inspired algorithms

- Something something two by two. 


`freelunch.adaptable`

- API for adaptable parameters / varying parameters and adaptable method selection


`freelunch.tech`

- Techniques / utility functions (i.e. bounding procedures, matrix representation tools etc) used by several optimisers

