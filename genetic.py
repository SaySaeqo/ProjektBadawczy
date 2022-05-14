import numpy as np

import network
from network import Network
from organism import *
from prints import *
from breedcrosses import *
import random
from mutations import *

# nazwa funkcji
# genetyczna_MetodaTworzeniaDzieci_RodzajMutacji
from utils import progress_bar


def genetic_func_mean_random(function, arg_num, domain_list, min_max, probe_num):
    generation_counter = 0
    organisms = create_population(arg_num, domain_list, probe_num)
    # print_generation(organisms,generation_counter,function)
    asses(organisms, function)

    # while not(Warunek(organisms)):
    while not (generation_counter == MAX_GENERATIONS):
        generation_counter += 1

        # krzyżyj organizmy(twórz ich dzieci)
        children = cross_breeds_avg(organisms, probe_num)
        # mutuj dzieci
        mutate_new_random(children, domain_list, arg_num)
        # ocen dzieci
        asses(children, function)
        # wybierz teraz najlepiej przystosowane
        organisms = select_best(organisms, children, min_max)

        print(generation_counter, '/', MAX_GENERATIONS)
    # print_generation(organisms,generation_counter,function)

    # ustawienie wyniku w zależności czego szukaliśmy
    if min_max == MAX:
        organisms = sorted(organisms, key=lambda x: x.ocena, reverse=True)
    elif min_max == MIN:
        organisms = sorted(organisms, key=lambda x: x.ocena, reverse=False)
    return organisms[0].data


def genetic_func_mean_change(function, arg_num, domain_list, min_max, probe_num):
    generation_counter = 0
    organisms = create_population(arg_num, domain_list, probe_num)
    # print_generation(organisms,generation_counter,function)
    asses(organisms, function)

    # while not(Warunek(organisms)):
    while not (generation_counter == MAX_GENERATIONS):
        generation_counter += 1

        # krzyżyj organizmy(twórz ich dzieci)
        children = cross_breeds_avg(organisms, probe_num)
        # mutuj dzieci
        mutate_change_little(children, domain_list, arg_num)
        # ocen dzieci
        asses(children, function)
        # wybierz teraz najlepiej przystosowane
        organisms = select_best(organisms, children, min_max)

        print(generation_counter, '/', MAX_GENERATIONS)
    # print_generation(organisms,generation_counter,function)

    # ustawienie wyniku w zależności czego szukaliśmy
    asses(organisms, function)
    if min_max == MAX:
        organisms = sorted(organisms, key=lambda x: x.ocena, reverse=True)
    elif min_max == MIN:
        organisms = sorted(organisms, key=lambda x: x.ocena, reverse=False)
    return organisms[0].data


def genetic_func_mean_gradient_change_litle(function, arg_num, domain_list, min_max, probe_num):
    generation_counter = 0
    organisms = create_population(arg_num, domain_list, probe_num)
    # print_generation(organisms, generation_counter, function)
    asses(organisms, function)

    # while not(Warunek(organisms)):
    while not (generation_counter == MAX_GENERATIONS):
        generation_counter += 1

        # krzyżyj organizmy(twórz ich dzieci)
        children = cross_breeds_avg(organisms, probe_num)
        # mutuj dzieci
        mutate_gradient_wise(children, function, domain_list, min_max)
        # ocen dzieci
        asses(children, function)
        # wybierz teraz najlepiej przystosowane
        organisms = select_tournament(organisms, children, min_max)

        print(generation_counter, '/', MAX_GENERATIONS)
    # print_generation(organisms,generation_counter,function)

    # ustawienie wyniku w zależności czego szukaliśmy
    asses(organisms, function)
    if min_max == MAX:
        organisms = sorted(organisms, key=lambda x: x.ocena, reverse=True)
    elif min_max == MIN:
        organisms = sorted(organisms, key=lambda x: x.ocena, reverse=False)
    return organisms[0].data


def genetic_func_mean_random4():
    pass


def net_genetic(net: Network, *args, **kwargs):
    """
    Correct function for neural network. Uses **genetic algorithm** to upgrade network weights and biases.

    :param inputs: list of network's inputs
        (when single input for network is a list, then it is list of lists)
    :param expected_outputs: list of expected outputs for each network's inputs in previous argument
    :return: void
    """
    # cost need to be minimal
    steps = []  # cost function values over iterations

    POPULATION_SIZE = kwargs.get("population_size", 50)
    ELITE_SIZE = kwargs.get("elite_size", 5)
    MUTATION_CHANCE = kwargs.get("mutation_chance", 0.15)

    # support for arguments type: list of tuples (input, ex_output)
    if len(args) == 1:
        args = args[0]
    elif len(args) == 2:
        inputs, ex_outputs = args
        args = list(zip(inputs, ex_outputs))

    population = [Network(net.model_shape) for _ in range(POPULATION_SIZE)]

    for iter in range(MAX_GENERATIONS):
        semi_step = []
        for input, ex_output in args:

            # selection
            input = np.matrix(input)
            ex_output = np.matrix(ex_output)
            costs = (np.sum(np.float_power(individual(input) - ex_output, 2))
                     for individual in population)
            population = [(individual, cost) for individual, cost in zip(population, costs)]
            population.sort(key=lambda a: a[1])
            elite = [individual for individual, cost in population[:ELITE_SIZE]]
            semi_step += [population[0][1]]  # for statistics
            population = [individual for individual, cost in population]

            # # crossover
            # children = []
            # for _ in range(ELITE_SIZE, POPULATION_SIZE):
            #     parents = random.sample(elite, 2)
            #     child = Network.create_empty(parents[0].model_shape)
            #     for i in range(child.nb_layers):
            #         child.layers[i] = parents[0].layers[i]
            #     children += [child]
            # population = elite + children

            # mutation
            for individual in population[ELITE_SIZE:]:
                layer = random.choice(individual.layers)
                w, b, _, _ = layer

                for i in range(len(w)):
                    for j in range(len(w[i])):
                        if random.random() < MUTATION_CHANCE:
                            w[i][j] += 0.1 * random.choice([-1, 1])
                for i in range(len(b)):
                    for j in range(len(b[i])):
                        if random.random() < MUTATION_CHANCE:
                            b[i][j] += 0.1 * random.choice([-1, 1])
                # weights_diff = np.random.rand(*w.shape) - 0.5
                # biases_diff = np.random.rand(*b.shape) - 0.5
                # w += weights_diff
                # b += biases_diff

        if kwargs.get("test_network_simple"):
            steps += semi_step
        else:
            steps += [sum(semi_step) / len(semi_step)]

        if MAX_GENERATIONS > 200:
            progress_bar(iter, MAX_GENERATIONS)

    net.layers = population[0].layers
    return steps
