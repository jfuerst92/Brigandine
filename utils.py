import numpy as np
import mlrose_hiive as mlrose
import random
import time
import matplotlib.pyplot as plt

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
