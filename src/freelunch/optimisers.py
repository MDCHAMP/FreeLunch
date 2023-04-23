"""Optimisation algorthms.

Each optimisation algorithm is implemented here as a subclass of the base optimiser class. 
"""

import numpy as np

from freelunch import tech
from freelunch.base import optimiser


# %% Optimisation classes


class RandomSearch(optimiser):
    """Sample at random from the search space."""

    name = "RandomSearch"
    tags = ["Stochastic"]
    hyper_definitions = {}
    hyper_defaults = {}

    def pre_loop(self):
        self.pos = tech.uniform_continuous_init(self.bounds, self.hypers["N"])
        self.fit = np.array([self.obj(x) for x in self.pos])

    def step(self):
        self.pre_loop()  # updating of global best is handled in obj


class DE(optimiser):
    """Differential Evolution.

    Efficient genetic inspired optimisation algorithm that approximates the finite difference operation.

    This class also acts as a base class for DE variants.

    Implemented as in: https://link.springer.com/article/10.1023/a:1008202821328

    Attributes:
        mutator (callable): Mutation strategy (see freelunch.tech for options)
        crossover (callable): Crossover strategy (see freelunch.tech for options)
        selector (callable): Selection strategy (see freelunch.tech for options)
    """

    name = "Differential Evolution"
    tags = ["continuous domain", "population based", "evolutionary"]
    hyper_definitions = {
        "F": "Mutation parameter (float in [0,1])",
        "Cr": "Crossover probability (float in [0,1])",
    }
    hyper_defaults = {
        "F": 0.5,
        "Cr": 0.1,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mutator = tech.rand_1
        self.crossover = tech.binary_crossover
        self.selector = tech.greedy_selection

    def pre_loop(self):
        self.pos = tech.uniform_continuous_init(self.bounds, self.hypers["N"])
        self.fit = np.array([self.obj(x) for x in self.pos])

    def step(self):
        newpos = self.mutator((self.pos, self.fit), self.hypers["F"])
        newpos = self.crossover(self.pos, newpos, self.hypers["Cr"])
        newpos = self.bounder(newpos, self.bounds)
        newfit = np.array([self.obj(x) for x in newpos])
        self.pos, self.fit = self.selector(
            (self.pos, self.fit), (newpos, newfit), return_idx=False
        )


class SADE(DE):
    """Self-Adapting Differential Evolution.

    Efficient genetic inspired optimisation algorithm that approximates the finite difference operation.

    Implemented as in: https://ieeexplore.ieee.org/document/1554904

    Attributes:
        mutators (list): Mutation strategies considered in the adaptation (see freelunch.tech for options)
    """

    name = "Self-Adapting Differential Evolution"
    tags = ["continuous domain", "population based", "evolutionary", "adaptive"]
    hyper_definitions = {
        "F_u": "Mutation parameter initial mean (float in [0,2])",
        "F_sig": "Mutation parameter initial standard deviation (float in [0,1])",
        "Cr_u": "Crossover probability initial mean (float in [0,1])",
        "Cr_sig": "Crossover probability initial standard deviation (float in [0,1])",
        "Lp": "Learning period",
    }
    hyper_defaults = {
        "F_u": 0.5,
        "F_sig": 0.3,
        "Cr_u": 0.2,
        "Cr_sig": 0.1,
        "Lp": 25,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mutators = [
            tech.rand_1,
            tech.rand_2,
            tech.best_1,
            tech.best_2,
            tech.current_2,
        ]

    def pre_loop(self):
        nm = len(self.mutators)
        self.pos = tech.uniform_continuous_init(self.bounds, self.hypers["N"])
        self.fit = np.array([self.obj(x) for x in self.pos])
        self.F = (self.hypers["F_u"], self.hypers["F_sig"])
        self.Cr = self.hypers["Cr_u"], self.hypers["Cr_sig"]
        self.F_wins, self.Cr_wins = [], []  # init tracking vars
        self.mut_probs = np.ones(nm) / nm
        self.mut_scores = (np.zeros(nm), np.zeros(nm))  # (hits, wins)

    def step(self):
        N, nm = self.hypers["N"], len(self.mutators)
        if self.gen % self.hypers["Lp"] == 0:  # Learning period complete?
            self.F = tech.normal_update(self.F, self.F_wins)
            self.Cr = tech.normal_update(self.Cr, self.Cr_wins)
            self.mut_probs = tech.update_selection_probs(self.mut_scores)
            self.F_wins, self.Cr_wins = [], []  # Reset tracking vars
            self.mut_scores = (np.zeros((nm,)), np.zeros((nm,)))  # (hits, wins)
        Fs = np.random.normal(*self.F, (N, 1))
        Crs = np.random.normal(*self.Cr, (N, 1))
        midx = np.random.choice(range(len(self.mutators)), size=N, p=self.mut_probs)
        newpos = self.pos.copy()
        for i, (m, f) in enumerate(zip(midx, Fs)):  # this loop is slow but unavoidable
            newpos[i] = self.mutators[m]((self.pos, self.fit), f, k=i)
        newpos = self.crossover(self.pos, newpos, Crs)
        newpos = self.bounder(newpos, self.bounds)
        newfit = np.array([self.obj(x) for x in newpos])
        (self.pos, self.fit), idx = self.selector(
            (self.pos, self.fit), (newpos, newfit), return_idx=True
        )
        self.F_wins.extend(Fs[idx])
        self.Cr_wins.extend(Fs[idx])
        self.mut_scores = tech.track_hits_wins(self.mut_scores, midx, idx)


class PSO(optimiser):
    """Particle Swarm Optimisation.

    Base class for all PSO variants.

    Implemented as in: https://ieeexplore.ieee.org/document/488968

    Note that the default hyperparameters here differ from those of the original paper above.
    """

    name = "Particle Swarm Optimisation"
    tags = ["Continuous domain", "Particle swarm"]
    hyper_definitions = {
        "I": "Inertia coefficients (np.array, I.shape=(2,))",
        "A": "Acceleration (np.array, I.shape=(2,))",
        "v0": "Velocity (ratio of bounds width)",
    }
    hyper_defaults = {
        "I": np.array([0.4, 0.3]),
        "A": np.array([0.01, 0.02]),
        "v0": 0,
    }

    def pre_loop(self):
        self.pos = tech.uniform_continuous_init(self.bounds, self.hypers["N"])
        self.vel = tech.uniform_continuous_init(
            (self.bounds - self.bounds.mean(1)[:, None]) * self.hypers["v0"],
            self.hypers["N"],
        )
        self.fit = np.array([self.obj(x) for x in self.pos])
        # Initial bookeeping
        self.local_best = self.pos.copy(), self.fit.copy()

    def step(self):
        N, G = self.hypers["N"], self.hypers["G"]
        # Calculate current inertia
        inertia = tech.lin_vary(self.hypers["I"], self.gen, G)
        # Update velocity
        self.vel = (
            inertia * self.vel
            + self.hypers["A"][0]
            * np.random.uniform((N,))
            * (self.local_best[0] - self.pos)
            + self.hypers["A"][1]
            * np.random.uniform((N,))
            * (self.global_best[0] - self.pos)
        )
        # Move
        self.pos += self.vel
        # Bounding
        self.pos = self.bounder(self.pos, self.bounds)
        # Eval fitness
        self.fit = np.array([self.obj(x) for x in self.pos])
        # Bookeeping post eval
        self.local_best = tech.greedy_selection(self.local_best, (self.pos, self.fit))

    def post_loop(self):
        self.pos, self.fit = self.local_best


class QPSO(PSO):
    """Quantum Particle Swarm Optimisation
    
    Implemented as in: @TJR TODO link to paper and report and differences
    """

    name = "Quantum Particle Swarm Optimisation"
    tags = ["Continuous domain", "Particle swarm"]
    hyper_definitions = {
        "alpha": "Contraction Expansion Coefficient (np.array, I.shape=(2,))",
        "v0": "Velocity (ratio of bounds width)",
    }
    hyper_defaults = {
        "alpha": np.array([1.0, 0.5]),
        "v0": 0,
    }

    def step(self):
        N, G = self.hypers["N"], self.hypers["G"]
        C = self.local_best[0].mean(0)
        D = len(C)
        alpha = tech.lin_vary(self.hypers["alpha"], self.gen, G)
        phi = np.random.uniform(size=(N, 1))
        u = np.random.uniform(size=(N, 1))
        self.pos = (
            phi * self.local_best[0]
            + (1 - phi) * self.global_best[0]
            + np.sign(np.random.normal(size=(N, D)))
            * alpha
            * np.abs(C - self.pos)
            * np.log(1 / u)
        )
        # Bounding
        self.pos = self.bounder(self.pos, self.bounds)
        # Eval fitness
        self.fit = np.array([self.obj(x) for x in self.pos])
        # Bookeeping post eval
        self.local_best = tech.greedy_selection(self.local_best, (self.pos, self.fit))


class PAO(PSO):
    """Particle Attractor Optimisation.

    (Rhymes with 'cow')

    Essentially a state-space implementation of a second order SDE with a number of weighted attractors.

    Implemented as in: PREPRINT TBA

    Note that a linearly varing value of q is implemented here.
    """

    name = "Particle Attractor Optimisation"
    tags = ["Continuous domain", "Particle swarm"]
    hyper_definitions = {
        "m": "Inertia coefficient (i.e mass)",
        "z": "Damping ratio",
        "k": "Stiffness parmaeters [k1, ..., kn]",
        "q": "Stochastic factor range [q0, qT]",
        "dt": "Timestep size",
        "v0": "Velocity (ratio of bounds width)",
    }
    hyper_defaults = {
        "m": 1,
        "z": 0.2,
        "k": np.array([1, 1]),
        "q": [1, 1],
        "dt": 1,
        "v0": 0,
    }

    def update_attractors(self):
        """Update the attraction centres and copmpute offset.

        Default behavour is to use local best and global best locations.

        This function can be overwritten to create custom variants of PAO with arbitrary attraction centres. When implementing a custom method care must be taken to calculate the coordinates of the overall offset (in the original coordinates) for each particle with shape (N,D).

        Returns:
            np.ndarray: Locations of the attraction centres
        """
        self.local_best = tech.greedy_selection(self.local_best, (self.pos, self.fit))
        # attractors
        attractors = np.stack(
            [self.local_best[0], np.ones_like(self.pos) * self.global_best[0]], axis=2
        ).transpose(2, 0, 1)
        # offset
        offset = np.dot(attractors.T, self.hypers["k"]).T / np.sum(self.hypers["k"])
        return offset

    def pre_loop(self):
        super().pre_loop()
        # init A matrix and Sig_L
        zet, m = self.hypers["z"], self.hypers["m"]
        wn2 = np.sum(self.hypers["k"]) / m
        A = np.array([[0, 1], [-wn2 / m, -2 * zet * np.sqrt(wn2) / m]])
        L = np.array([[0], [1]])
        Phi = tech.expm(
            np.block([[A, L @ L.T], [np.zeros_like(A), -A.T]]) * self.hypers["dt"]
        )
        self.Adt = Phi[:2, :2]
        self.Sig_L = np.linalg.cholesky(Phi[:2, 2:] @ self.Adt.T)
        # preallocate swam states in transformed coordinates
        self.Xprime = np.stack((self.pos, self.vel), axis=2)

    def step(self):
        N, D = self.pos.shape[0], self.bounds.shape[0]
        # update attraction centres
        self.offset = self.update_attractors()
        # compute scaling factor of noise for each state
        q = tech.lin_vary(self.hypers["q"], self.gen, self.hypers["G"]) * (
            np.abs(self.local_best[0] - self.global_best[0])
        )
        # recover swarm states in generalised coordinates
        self.Xprime[..., 0] = self.pos - self.offset
        # move the swarm in generalised coordinates
        draws = np.random.standard_normal(size=(N, D, 2, 1))
        self.Xprime = (
            self.Adt @ self.Xprime[..., None]
            + (q[..., None, None] * self.Sig_L) @ draws
        )[..., 0]
        # Recover swarm in original coordinates
        self.pos = self.Xprime[..., 0] + self.offset
        self.vel = self.Xprime[..., 1]
        self.pos = self.bounder(self.pos, self.bounds)
        self.fit = np.array([self.obj(x) for x in self.pos])


# class ABC(optimiser):
#     '''
#     Artificial Bee Colony
#     '''
#     name = 'Artificial Bee Colony Optimisation'
#     tags = ['Continuous domain', 'Particle swarm', 'Utterly terrible']
#     hyper_definitions = {
#         'N': 'Population size (int)',
#         'G': 'Number of generations (int)',
#     }
#     hyper_defaults = {
#         'N': 100,
#         'G': 100,
#         'limit': 5,
#     }

#     def pre_loop(self):
#         self.pos = tech.uniform_continuous_init(self.bounds, self.hypers['N'])
#         self.fit = np.array([self.obj(x) for x in self.pos])
#         self.best_fit = np.min(self.fit)
#         self.best_pos = self.pos[np.argmin(self.fit)]
#         self.stalls = np.zeros((self.hypers["N"]))
#         self.hypers["N_employed"] = self.hypers["N"]//2
#         self.hypers["N_onlookers"] = self.hypers["N"] - self.hypers["N_employed"]

#     @staticmethod
#     def crossover_points(pos): # TODO This should be in tech
#         """ABC Crossover Location

#         Args:
#             pos (np.ndarray): postitions to update

#         Returns:
#             np.ndarray: crossover points (x_{ik} - x_{jk})
#         """

#         N = pos.shape[0]
#         crossover_point = pos.copy()
#         d = np.random.randint(0, pos.shape[1], size=(N,1))
#         rand_idx = np.arange(N) + np.random.randint(1, N, size=(N,))
#         rand_idx[rand_idx>=N] -= N
#         mask = np.tile(np.arange(pos.shape[1]),(N,1)) == d
#         crossover_point[mask] -= pos[rand_idx][mask]

#         return crossover_point

#     def step(self):

#         # The employed bees (good capitalist bees)
#         employed = self.pos[:self.hypers["N_employed"]]
#         employed_fit = self.fit[:self.hypers["N_employed"]]
#         crossover_point = self.crossover_points(employed)
#         new_candidates = employed + np.random.uniform(-1,1,size=(self.hypers["N_employed"],1))*crossover_point

#         # Greedy Selection...
#         new_fit = np.array([self.obj(x) for x in new_candidates])
#         tech.greedy_selection(employed_fit, new_fit, employed, new_candidates)

#         # "Fitness" of each employed bee
#         fit_xm = np.ones_like(employed_fit)
#         fit_xm[employed_fit >= 0] = 1/(1+employed_fit[employed_fit >= 0])
#         fit_xm[employed_fit < 0] = 1+np.abs(employed_fit[employed_fit < 0])

#         # Enter the onlooker bee phase (bad unproductive bees)

#         # Pinwheel onlookers choosing one of the employees to take credit for...
#         idx = np.random.choice(
#             np.arange(self.hypers["N_employed"]),
#             size=self.hypers["N_onlookers"],
#             p=fit_xm/np.sum(fit_xm),
#             replace=True)
#         onlookers = employed[idx]
#         onlookers_fit = employed_fit[idx]

#         # Onlooker selections
#         crossover_point = self.crossover_points(onlookers)
#         new_candidates = onlookers + np.random.uniform(-1,1,size=(self.hypers["N_onlookers"],1))*crossover_point

#         # Greedy Selection...
#         new_fit = np.array([self.obj(x) for x in new_candidates])
#         tech.greedy_selection(onlookers_fit, new_fit, onlookers, new_candidates)

#         # Cleaning up
#         new_pop = np.concatenate((employed, onlookers),axis=0)
#         new_fit = np.concatenate((employed_fit, onlookers_fit),axis=0)
#         self.stalls[np.all(self.pos == new_pop, axis=1)] += 1
#         self.pos = new_pop
#         self.fit = new_fit

#         if np.any(self.fit < self.best_fit):
#             self.best_fit = np.min(self.best_fit)
#             self.best_pos = self.pos[np.argmin(self.fit)]
