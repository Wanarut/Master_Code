import pandas as pd
from timeit import default_timer as timer
from mlxtend.preprocessing import TransactionEncoder
import random as rn
import multiprocessing as mp
from functools import partial
import numpy as np

# Set general parameters
starting_population_size = 3
maximum_generation = 5
minimum_population_size = 2
maximum_population_size = 3
print_interval = 1
xover_rate = 0.5
mutat_rate = 0.5
n_mutation = 1
# multi-processing
slave_core = int(mp.cpu_count()-1)

# file_name = ['dataset/T10I4D100K.dat', ' ']
# file_name = ['dataset/chess.dat', ' ']
# file_name = ['dataset/store_data.csv', ',']
file_name = ['dataset/test_data.csv', ',']


def main():
    # # pre-processing data
    dataset = []
    lines = open(file_name[0], 'r')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        dataset.append(line.split(file_name[1]))
    # print(dataset[0])
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    print(df.head())

    start = timer()
    print('Start POGA Optimization')
    rules = []
    rules = POGA(df)
    # try:
    #     rules = POGA(df)
    # except:
    #     print('POGA Fail')
    used_time = timer()-start

    if len(rules) > 0:
        print(rules.head())
        rules.to_csv('mlxtend_output.csv', index=False, header=True)

    print('There are', df.shape, 'transections')
    print('Found', len(rules), 'rules')
    print('Use', pd.to_timedelta(used_time, unit='s'), 'second\n')


def POGA(dataset):
    rn.seed(69)
    population = create_population(dataset, starting_population_size)

    for generation in range(maximum_generation):
        # Breed
        print('P size', population.shape)
        population = breed_population(population)
        print('Q size', population.shape)

        scores = score_population(True, dataset, population)
        
    return population


def create_population(df, population_size):
    population = df.sample(n=population_size)
    # print(population)
    population = population.to_numpy()
    # print(population)
    # exit()
    return population

# Crossover
def breed_population(population):
    """
    Create child population by repetedly calling breeding function (two parents
    producing two children), applying genetic mutation to the child population,
    combining parent and child population, and removing duplicatee chromosomes.
    """
    # Create an empty list for new population
    new_population = []
    population_size = population.shape[0]
    # Create new population generating two children at a time
    for _ in range(int(population_size*xover_rate)):
        parent_1_loc = rn.randint(0, population_size-1)
        parent_2_loc = rn.randint(0, population_size-1)
        child_1, child_2 = twopoint_xover(population, parent_1_loc, parent_2_loc)
        new_population.append(child_1)
        new_population.append(child_2)

    # Add the child population to the parent population
    # In this method we allow parents and children to compete to be kept
    new_population = np.array(new_population)
    new_population = randomly_mutate_population(new_population)
    population = np.vstack((population, new_population))
    population = np.unique(population, axis=0)

    return population


def twopoint_xover(population, parent_1_loc, parent_2_loc):
    """
    Combine two parent chromsomes by crossover to produce two children.
    """
    parent_1 = population[parent_1_loc]
    parent_2 = population[parent_2_loc]

    # Get length of chromosome
    chromosome_length = len(parent_1)

    # Pick crossover point, avoding ends of chromsome
    xover_points = [rn.randint(1, chromosome_length-2), rn.randint(1, chromosome_length-2)]

    # Create children.
    child_1 = parent_1.copy()
    child_2 = parent_2.copy()
    for point in xover_points:
        child_1, child_2 = singlepoint_xover(child_1, child_2, point)

    # Return children
    return child_1, child_2


def singlepoint_xover(A, B, x):
    A_new = np.append(A[:x], B[x:])
    B_new = np.append(B[:x], A[x:])
    return A_new, B_new

# Mutation
def randomly_mutate_population(population):
    """
    Randomly mutate population with a given solution gene mutation probability.
    """
    population_size = population.shape[0]
    chromosome_length = population.shape[1]
    # Apply random mutation through each row (solution)
    for i in range(int(population_size*mutat_rate)):
        j = rn.randint(0, chromosome_length-1)
        population[i, j] = not population[i, j]

    # Return mutation population
    return population


def score_population(display, dataset, population):
    """
    Loop through all objectives and request score/fitness of population.
    """
    population_size = population.shape[0]
    scores = np.zeros((population_size, 3), int)
    show_interval = int(population_size/50) + 1
    
    for i in range(population_size):
        # if display and i % show_interval == 0 :
        #     print('-', end='', flush=True)
        scores[i, 0] = calculate_support_fitness(population[i], dataset)
        scores[i, 1] = calculate_confidence_fitness(population[i], dataset)
        scores[i, 2] = calculate_lift_fitness(population[i], dataset)
    
    return scores
    # return_dict[index] = scores


def calculate_support_fitness(individual, dataset):
    # individual => dataset
    result = ~(individual) | dataset
    freq_X = result[result.all(axis=1)].shape[0]
    support = freq_X/dataset.shape[0]
    
    return support

def calculate_confidence_fitness(individual, dataset):
    # individual => dataset
    result = ~(individual) | dataset
    freq_X = result[result.all(axis=1)].shape[0]
    support = freq_X/dataset.shape[0]
    
    return support

if __name__ == '__main__':
    main()
