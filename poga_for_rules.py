import pandas as pd
from timeit import default_timer as timer
from mlxtend.preprocessing import TransactionEncoder
import random as rn
import multiprocessing as mp
from functools import partial
import numpy as np
import sys
epsilon = sys.float_info.epsilon

# Set general parameters
starting_population_size = 10
maximum_generation = 5
minimum_population_size = 8
maximum_population_size = 10
print_interval = 1
xover_rate = 0.5
mutat_rate = 0.5
n_mutation = 1
# multi-processing
slave_core = int(mp.cpu_count()-1)

# file_name = ['dataset/T10I4D100K.dat', ' ']
# file_name = ['dataset/chess.dat', ' ']
file_name = ['dataset/store_data.csv', ',']
# file_name = ['dataset/test_data.csv', ',']


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
    # exit()
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
    # rn.seed(69)
    population = create_population(dataset, starting_population_size)
    population = randomly_mutate_population(population)

    for generation in range(maximum_generation):
        mstart = timer()
        # Breed
        population = breed_population(population)

        if generation % print_interval == 0:
            print('Generation (out of %i): gen %i' % (maximum_generation, generation + 1), end='', flush=True)
            display = True
        else :
            display = False
        scores = score_population(display, dataset, population)
        population, scores = build_pareto_population(population, scores, minimum_population_size, maximum_population_size)

        # time per population
        tpg = timer()-mstart
        if generation % print_interval == 0:
            print('> use' , pd.to_timedelta(tpg, unit='s'), '/gen -> estimated time left', pd.to_timedelta((maximum_generation-generation)*tpg, unit='s'))
    
    population_ids = np.arange(population.shape[0]).astype(int)
    pareto_front = identify_pareto(scores, population_ids)
    population = population[pareto_front, :]
    scores = scores[pareto_front]
    
    order = np.argsort(scores[:, 2])
    population = population[order]
    scores = scores[order]

    for score in scores:
        print(score)
    exit()
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
    # new_population = randomly_mutate_population(new_population)
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
    scores = np.zeros((population_size, 3), float)
    show_interval = int(population_size/50) + 1
    
    for i in range(population_size):
        if display and i % show_interval == 0 :
            print('-', end='', flush=True)
        sup_XY, sup_X, sup_Y = calculate_support_fitness(population[i], dataset)
        scores[i, 0] = sup_XY
        scores[i, 1] = sup_XY/sup_X
        scores[i, 2] = sup_XY/(sup_X*sup_Y)
    
    return scores
    # return_dict[index] = scores


def calculate_support_fitness(individual, dataset):
    # individual => dataset
    result = ~(individual) | dataset
    freq_XY = result[result.all(axis=1)].shape[0]+epsilon
    loc_Y = np.argmax(individual==True)

    individual_X = individual.copy()
    individual_X[loc_Y] = False
    individual_Y = individual.copy()
    individual_Y = individual_Y & False
    individual_Y[loc_Y] = True

    result = ~(individual_X) | dataset
    freq_X = result[result.all(axis=1)].shape[0]+epsilon
    result = ~(individual_Y) | dataset
    freq_Y = result[result.all(axis=1)].shape[0]+epsilon
    
    N = dataset.shape[0]
    sup_XY, sup_X, sup_Y = freq_XY/N, freq_X/N, freq_Y/N
    return sup_XY, sup_X, sup_Y

# Pareto front
def build_pareto_population(population, scores, minimum_population_size, maximum_population_size):
    """
    As necessary repeats Pareto front selection to build a population within
    defined size limits. Will reduce a Pareto front by applying crowding 
    selection as necessary.    
    """
    unselected_population_ids = np.arange(population.shape[0])
    all_population_ids = np.arange(population.shape[0])
    pareto_front = []
    while len(pareto_front) < minimum_population_size:
        temp_pareto_front = identify_pareto(
            scores[unselected_population_ids, :], unselected_population_ids)

        # Check size of total parteo front.
        # If larger than maximum size reduce new pareto front by crowding
        combined_pareto_size = len(pareto_front) + len(temp_pareto_front)
        if combined_pareto_size > maximum_population_size:
            number_to_select = combined_pareto_size - maximum_population_size
            selected_solutions = (reduce_by_crowding(
                scores[temp_pareto_front], number_to_select))
            temp_pareto_front = temp_pareto_front[selected_solutions]

        # Add latest pareto front to full Pareto front
        pareto_front = np.hstack((pareto_front, temp_pareto_front))

        # Update unselected population ID by using sets to find IDs in all
        # ids that are not in the selected front
        unselected_set = set(all_population_ids) - set(pareto_front)
        unselected_population_ids = np.array(list(unselected_set))

    index = pareto_front.astype(int)
    population = population[index]
    scores = scores[index]
    return population, scores


def identify_pareto(scores, population_ids):
    """
    Identifies a single Pareto front, and returns the population IDs of
    the selected solutions.
    """
    population_size = scores.shape[0]
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]


def reduce_by_crowding(scores, number_to_select):
    """
    This function selects a number of solutions based on tournament of
    crowding distances. Two members of the population are picked at
    random. The one with the higher croding distance is always picked
    """
    population_ids = np.arange(scores.shape[0])

    crowding_distances = calculate_crowding(scores)

    picked_population_ids = np.zeros((number_to_select))

    picked_scores = np.zeros((number_to_select, len(scores[0, :])))

    for i in range(number_to_select):

        population_size = population_ids.shape[0]

        fighter1ID = rn.randint(0, population_size - 1)

        fighter2ID = rn.randint(0, population_size - 1)

        # If fighter # 1 is better
        if crowding_distances[fighter1ID] >= crowding_distances[fighter2ID]:

            # add solution to picked solutions array
            picked_population_ids[i] = population_ids[fighter1ID]

            # Add score to picked scores array
            picked_scores[i, :] = scores[fighter1ID, :]

            # remove selected solution from available solutions
            population_ids = np.delete(population_ids, (fighter1ID), axis=0)

            scores = np.delete(scores, (fighter1ID), axis=0)

            crowding_distances = np.delete(crowding_distances, (fighter1ID), axis=0)
        else:
            picked_population_ids[i] = population_ids[fighter2ID]

            picked_scores[i, :] = scores[fighter2ID, :]

            population_ids = np.delete(population_ids, (fighter2ID), axis=0)

            scores = np.delete(scores, (fighter2ID), axis=0)

            crowding_distances = np.delete(
                crowding_distances, (fighter2ID), axis=0)

    # Convert to integer
    picked_population_ids = np.asarray(picked_population_ids, dtype=int)

    return (picked_population_ids)


def calculate_crowding(scores):
    """
    Crowding is based on a vector for each solution
    All scores are normalised between low and high. For any one score, all
    solutions are sorted in order low to high. Crowding for chromsome x
    for that score is the difference between the next highest and next
    lowest score. Total crowding value sums all crowding for all scores
    """

    population_size = len(scores[:, 0])
    number_of_scores = len(scores[0, :])

    # create crowding matrix of population (row) and score (column)
    crowding_matrix = np.zeros((population_size, number_of_scores))

    # normalise scores (ptp is max-min)
    normed_scores = (scores - scores.min(0)) / 1+scores.ptp(0)

    # calculate crowding distance for each score in turn
    for col in range(number_of_scores):
        crowding = np.zeros(population_size)

        # end points have maximum crowding
        crowding[0] = 1
        crowding[population_size - 1] = 1

        # Sort each score (to calculate crowding between adjacent scores)
        sorted_scores = np.sort(normed_scores[:, col])

        sorted_scores_index = np.argsort(normed_scores[:, col])

        # Calculate crowding distance for each solution
        crowding[1:population_size - 1] = (sorted_scores[2:population_size] - sorted_scores[0:population_size - 2])

        # resort to orginal order (two steps)
        re_sort_order = np.argsort(sorted_scores_index)
        sorted_crowding = crowding[re_sort_order]

        # Record crowding distances
        crowding_matrix[:, col] = sorted_crowding

    # Sum crowding distances of each score
    crowding_distances = np.sum(crowding_matrix, axis=1)

    return crowding_distances


if __name__ == '__main__':
    main()
