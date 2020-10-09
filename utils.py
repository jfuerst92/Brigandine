import numpy as np
import mlrose_hiive as mlrose
import random
import time
import matplotlib.pyplot as plt
from mlrose_hiive.fitness._discrete_peaks_base import _DiscretePeaksBase

def knapsack_generator(length):
    values = []
    weights = []
    for i in range(length):
        values = random.sample(range(1, length+1), length)
        weights = random.sample(range(1, length+1), length)
    return values, weights

def fit_iteration_test(problem, iters, algorithm, args, plot=False, alg_name="", prob_name=""):
    args['max_iters'] = iters
    args['max_attempts'] = iters
    if iters == np.inf:
        args['max_attempts'] = 100
    return algorithm(**args)


def time_iteration_test(iter_range, problem, algorithm, args, plot=False, alg_name="", prob_name=""):
    args['problem'] = problem
    times = []
    for max_iters in iter_range:
        start_time = time.time()
        args['max_iters'] = max_iters
        args['max_attempts'] = max_iters
        state, fitness, curve = algorithm(**args)
        time_elapsed = time.time() - start_time
        times.append(time_elapsed)
    if plot:
        plt.plot(param_range, fits, label=alg_name)
        plt.xlabel (param_name)
        plt.ylabel ('fit')
        plt.legend()
        title = "" + alg_name + " " + param_name + " fits " + prob_name
        plt.title(title)
        plt.grid(True, linestyle='-.')
        plt.show()
    return times

def test_param(param_name, param_range, problem, algorithm, args, plot=False, alg_name="", prob_name=""):
    args['problem'] = problem
    fits = []
    best_fit = 0
    best_arg_val = 0
    if param_name == 'decay':
        #this is SA, we need to define schedule
        for p in param_range:
            print(p)
            args['schedule'] = mlrose.GeomDecay(init_temp=10, decay=p, min_temp=0.001)
            state, fitness, curve = algorithm(**args)
            fits.append(fitness)
            if fitness > best_fit:
                best_fit = fitness
                best_arg_val = p
    else:
        for p in param_range:
            args[param_name] = p
            state, fitness, curve = algorithm(**args)
            fits.append(fitness)
            if fitness > best_fit:
                best_fit = fitness
                best_arg_val = p
    print("best value for ", param_name, ": ", best_arg_val)
    print("best fit: ", best_fit)
    if plot:
        plt.plot(param_range, fits, label=alg_name)
        plt.xlabel (param_name)
        plt.ylabel ('fit')
        plt.legend()
        title = "" + alg_name + " " + param_name + " fits " + prob_name
        plt.title (title)
        plt.grid(True, linestyle='-.')
        plt.show()
    return best_arg_val, best_fit


class OneMaxEvalCount:
    """Fitness function for One Max optimization problem. Evaluates the
    fitness of an n-dimensional state vector
    :math:`x = [x_{0}, x_{1}, \\ldots, x_{n-1}]` as:
    .. math::
        Fitness(x) = \\sum_{i = 0}^{n-1}x_{i}
    Example
    -------
    .. highlight:: python
    .. code-block:: python
        >>> import mlrose_hiive
        >>> import numpy as np
        >>> fitness = mlrose_hiive.OneMax()
        >>> state = np.array([0, 1, 0, 1, 1, 1, 1])
        >>> fitness.evaluate(state)
        5
    Note
    -----
    The One Max fitness function is suitable for use in either discrete or
    continuous-state optimization problems.
    """

    def __init__(self):

        self.prob_type = 'either'
        self.num_evals = 0

    def evaluate(self, state):
        """Evaluate the fitness of a state vector.
        Parameters
        ----------
        state: array
            State array for evaluation.
        Returns
        -------
        fitness: float
            Value of fitness function.
        """

        fitness = np.sum(state)
        self.num_evals += 1
        #print(self.num_evals)
        return fitness

    def get_prob_type(self):
        """ Return the problem type.
        Returns
        -------
        self.prob_type: string
            Specifies problem type as 'discrete', 'continuous', 'tsp'
            or 'either'.
        """
        return self.prob_type

class FourPeaksEvalCount(_DiscretePeaksBase):
    """Fitness function for Four Peaks optimization problem. Evaluates the
    fitness of an n-dimensional state vector :math:`x`, given parameter T, as:
    .. math::
        Fitness(x, T) = \\max(tail(0, x), head(1, x)) + R(x, T)
    where:
    * :math:`tail(b, x)` is the number of trailing b's in :math:`x`;
    * :math:`head(b, x)` is the number of leading b's in :math:`x`;
    * :math:`R(x, T) = n`, if :math:`tail(0, x) > T` and
      :math:`head(1, x) > T`; and
    * :math:`R(x, T) = 0`, otherwise.
    Parameters
    ----------
    t_pct: float, default: 0.1
        Threshold parameter (T) for Four Peaks fitness function, expressed as
        a percentage of the state space dimension, n (i.e.
        :math:`T = t_{pct} \\times n`).
    Example
    -------
    .. highlight:: python
    .. code-block:: python
        >>> import mlrose_hiive
        >>> import numpy as np
        >>> fitness = mlrose_hiive.FourPeaks(t_pct=0.15)
        >>> state = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        >>> fitness.evaluate(state)
        16
    References
    ----------
    De Bonet, J., C. Isbell, and P. Viola (1997). MIMIC: Finding Optima by
    Estimating Probability Densities. In *Advances in Neural Information
    Processing Systems* (NIPS) 9, pp. 424–430.
    Note
    ----
    The Four Peaks fitness function is suitable for use in bit-string
    (discrete-state with :code:`max_val = 2`) optimization problems *only*.
    """

    def __init__(self, t_pct=0.1):

        self.t_pct = t_pct
        self.prob_type = 'discrete'
        self.num_evals = 0

        if (self.t_pct < 0) or (self.t_pct > 1):
            raise Exception("""t_pct must be between 0 and 1.""")

    def evaluate(self, state):
        """Evaluate the fitness of a state vector.
        Parameters
        ----------
        state: array
            State array for evaluation.
        Returns
        -------
        fitness: float.
            Value of fitness function.
        """
        _n = len(state)
        _t = np.ceil(self.t_pct*_n)

        # Calculate head and tail values
        tail_0 = self.tail(0, state)
        head_1 = self.head(1, state)

        # Calculate R(X, T)
        if (tail_0 > _t and head_1 > _t):
            _r = _n
        else:
            _r = 0

        # Evaluate function
        fitness = max(tail_0, head_1) + _r
        self.num_evals += 1
        return fitness

    def get_prob_type(self):
        """ Return the problem type.
        Returns
        -------
        self.prob_type: string
            Specifies problem type as 'discrete', 'continuous', 'tsp'
            or 'either'.
        """
        return self.prob_type


class KnapsackEvalCount:

    def __init__(self, weights, values, max_weight_pct=0.35, max_item_count=1, multiply_by_max_item_count=False):

        self.weights = weights
        self.values = values
        count_multiplier = max_item_count if multiply_by_max_item_count else 1.0
        self._w = np.ceil(np.sum(self.weights) * max_weight_pct * count_multiplier)
        self.prob_type = 'discrete'
        self.num_evals = 0

        if len(self.weights) != len(self.values):
            raise Exception("""The weights array and values array must be"""
                            + """ the same size.""")

        if min(self.weights) <= 0:
            raise Exception("""All weights must be greater than 0.""")

        if min(self.values) <= 0:
            raise Exception("""All values must be greater than 0.""")

        if max_item_count <= 0:
            raise Exception("""max_item_count must be greater than 0.""")

        if max_weight_pct <= 0:
            raise Exception("""max_weight_pct must be greater than 0.""")

    def evaluate(self, state):
        """Evaluate the fitness of a state vector.
        Parameters
        ----------
        state: array
            State array for evaluation. Must be the same length as the weights
            and values arrays.
        Returns
        -------
        fitness: float
            Value of fitness function.
        """

        if len(state) != len(self.weights):
            raise Exception("""The state array must be the same size as the"""
                            + """ weight and values arrays.""")

        # Calculate total weight and value of knapsack
        total_weight = np.sum(state*self.weights)
        total_value = np.sum(state*self.values)

        # Allow for weight constraint
        if total_weight <= self._w:
            fitness = total_value
        else:
            fitness = 0
        self.num_evals += 1
        return fitness

    def get_prob_type(self):
        """ Return the problem type.
        Returns
        -------
        self.prob_type: string
            Specifies problem type as 'discrete', 'continuous', 'tsp'
            or 'either'.
        """
        return self.prob_type
