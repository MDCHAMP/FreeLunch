"""Benchmarks utilities for testing optimisers.

Each benchmarks derives from the benchmarks base class.
"""

import numpy as np

# %% Base class


class benchmark:
    """Base class for benchmark functions.

    Attributes:
        bounds (np.ndarray): Default bounds for the optimisation problem.
        optimum (np.ndarray): Location of optimum
        f0 (float): Value of benchmark function at the optimum.
    """

    def __init__(self, n=2):
        """Initialise the benchmark.

        Args:
            n (int, optional): Problem dimension. Defaults to 2.
        """
        self.n = n
        self.bounds = self._get_bounds()
        self.optimum = self._get_optimum()

    def __call__(self, x):
        """Evaluate the benchmark at location x.

        Args:
            x (np.ndarray): Location at which to evaluate the benchmark.

        Returns:
            float: Value of the benchmark at position x.
        """
        pass

    def _get_bounds(self):
        """Get default bounds for the optimisation problem."""
        pass

    def _get_optimum(self):
        """Get true position of optimum for the optimisation problem."""
        pass


# %% Benchmark functions


class ackley(benchmark):
    """Ackley benchmark function.

    Ackley function implemented in n dimensions.

    For details see: https://www.sfu.ca/~ssurjano/ackley.html
    """

    f0 = 0
    a, b, c = 20, 0.2, 2 * np.pi

    def __call__(self, x):
        t1 = -self.a * np.exp(-self.b * (1 / len(x)) * np.sum(x**2))
        t2 = -np.exp(1 / len(x) * np.sum(np.cos(self.c * x)))
        t3 = self.a + np.exp(1)
        return t1 + t2 + t3

    def _get_bounds(self):
        return np.array([[-32.768, 32.768]] * self.n)

    def _get_optimum(self):
        return np.zeros(self.n)


class exponential(benchmark):
    """Exponential benchmark function.

    Exponential function implemented in n dimensions.

    Note that this function is the negative of the traditional maximisation problem

    For details see: https://al-roomi.org/benchmarks/unconstrained/n-dimensions/168-exponential-function
    """

    f0 = -1
    a = 0.5

    def __call__(self, x):
        return -np.exp(-self.a * np.sum(x**2))

    def _get_bounds(self):
        return np.array([[-1, 1]] * self.n)

    def _get_optimum(self):
        return np.zeros(self.n)


class sphere(benchmark):
    """Sphere benchmark function.

    Sphere function implemented in n dimensions.

    For details see: https://www.sfu.ca/~ssurjano/spheref.html
    """

    f0 = 0

    def __call__(self, x):
        return np.sum(x**2)

    def _get_bounds(self):
        return np.array([[-5.12, 5.12]] * self.n)

    def _get_optimum(self):
        return np.zeros(self.n)


class happycat(benchmark):
    """Happycat benchmark function.

    Happycat function implemented in n dimensions.

    For details see: https://homepages.fhv.at/hgb/New-Papers/PPSN12_BF12.pdf
    """

    f0 = 0
    a = 1 / 8

    def __call__(self, x):
        norm = np.sum(x**2)
        t1 = ((norm - self.n) ** 2) ** (self.a)
        t2 = (1 / self.n) * (0.5 * norm + np.sum(x)) + 0.5
        return t1 + t2

    def _get_bounds(self):
        return np.array([[-2, 2]] * self.n)

    def _get_optimum(self):
        return -np.ones(self.n)


class periodic(benchmark):
    """Periodic benchmark function.

    Periodic function implemented in n dimensions.

    For details see: https://al-roomi.org/benchmarks/unconstrained/2-dimensions/158-price-s-function-no-2
    """

    f0 = 0.9

    def __call__(self, x):
        t1 = np.sum(np.sin(x) ** 2) + 1
        t2 = -0.1 * np.exp(-np.sum(x**2))
        return t1 + t2

    def _get_bounds(self):
        return np.array([[-10, 10]] * self.n)

    def _get_optimum(self):
        return np.zeros(self.n)
