from env import *
from evolutionary_computation import *
import numpy as np
import pickle
import os
from copy import deepcopy
import argparse

def experiment(params):
    env = EAGymEnvWrapper(params.env_name)

    """
    Evolutionary Algorithm
    Let's try using the one we built in Evolutionary Computation
    """

    experiment_results = {}
    solutions_results = {}
    diversity_results = {}

    num_runs = params.runs
    total_generations = params.total_generations
    num_elements_to_mutate = 1
    bit_string_length = env.genome_size
    num_parents = 10#20
    num_children = 10#20

    np.random.seed(50)

    fitness_records = np.empty((num_runs, total_generations))
    diversity_records = np.empty((num_runs, total_generations))
    solution_records = []

    for i in range(num_runs):
        print("Run " + str(i))
        f, s, d = evolutionary_algorithm(
            fitness_function = env.get_fitness,
            total_generations = total_generations,
            num_parents = num_parents,
            num_children = num_children,
            continuous = args.continuous,
            genome_length = bit_string_length,
            num_elements_to_mutate = num_elements_to_mutate,
            crossover = False,
            restart_every = 0,
            downhill_prob = 0.01,
            tournament_selection = False,
            novelty_selection = False,
            # novelty_k = novelty_k,
            # novelty_selection_prop = 0,
            # max_archive_length = 100,
            return_details = False)
        fitness_records[i] = f
        diversity_records[i] = d
        solution_records.append(s)

    experiment_results[params.env_name] = deepcopy(fitness_records)
    solutions_results[params.env_name] = deepcopy(solution_records)
    diversity_results[params.env_name] = deepcopy(diversity_records)

    os.makedirs("results_" + params.env_name.lower(), exist_ok = True)

    with open("results_" + params.env_name.lower() + "/" + \
        params.env_name.lower() + "_experiment_results.pkl",
        'wb') as filehandler:
        pickle.dump(experiment_results, filehandler)

    with open("results_" + params.env_name.lower() + "/" + \
        params.env_name.lower() + "_diversity_results.pkl",
        'wb') as filehandler:
        pickle.dump(diversity_results, filehandler)

    with open("results_" + params.env_name.lower() + "/" + \
        params.env_name.lower() + "_solutions_results.pkl",
        'wb') as filehandler:
        pickle.dump(solutions_results, filehandler)

    experiment_results = {}
    diversity_results = {}

    with open("results_" + params.env_name.lower() + "/" + \
        params.env_name.lower() + "_experiment_results.pkl", 'rb') as filehandler:
        experiment_results = pickle.load(filehandler)

    with open("results_" + params.env_name.lower() + "/" + \
        params.env_name.lower() + "_diversity_results.pkl", 'rb') as filehandler:
        diversity_results = pickle.load(filehandler)

    plot_mean_and_bootstrapped_ci_over_time(experiment_results,
        n_samples = 2000,
        figure_path = "results_" + params.env_name.lower() + "/" + \
            params.env_name.lower() + "_experiment_results.png")
    plot_mean_and_bootstrapped_ci_over_time(diversity_results,
        name = "Diversity Over Generations",
        y_label = "Diversity",
        n_samples = 2000,
        figure_path = "results_" + params.env_name.lower() + "/" + \
            params.env_name.lower() + "_diversity_results.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
                        # prog='ProgramName',
                        # description='What the program does',
                        # epilog='Text at the bottom of help')
    parser.add_argument('-env', "--env_name",
                        dest = 'env_name',
                        action = 'store',
                        nargs = '?',
                        choices = ["CartPole-v1", "MountainCar-v0"],
                        default = "CartPole-v1",
                        const = "CartPole-v1",
                        )
    parser.add_argument('-r', "--runs",
                        dest = "runs",
                        action = 'store',
                        type = int,
                        default = 100)
    parser.add_argument('-g',  "--gens",
                        dest = "total_generations",
                        action = 'store',
                        type = int,
                        default = 1000)
    parser.add_argument('-c',  "--continuous",
                        dest = "continuous",
                        action = 'store_true')
    args = parser.parse_args()
    experiment(args)
