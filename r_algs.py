import numpy as np
import mlrose_hiive as mlrose

def main():
    OUTPUT_DIRECTORY = './output'
    SEED = 13
    experiment_name = 'example_experiment'
    values = [1, 4, 5, 7]
    weights = [1, 3 ,4 ,5]

    max_weight_pct = 0.6
    fitness_fn = mlrose.Knapsack(weights, values, max_weight_pct)
    problem = mlrose.DiscreteOpt(length=len(weights),
                                 fitness_fn=fitness_fn
                                )

    sa = mlrose.SARunner(problem=problem,
                  experiment_name=experiment_name,
                  output_directory=OUTPUT_DIRECTORY,
                  seed=SEED,
                  iteration_list=2 ** np.arange(14),
                  max_attempts=5000,
                  temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000])
    # the two data frames will contain the results
    df_run_stats, df_run_curves = sa.run()

if __name__ == "__main__":
    main()
