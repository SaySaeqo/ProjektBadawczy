import copy

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

    params = NConst.instance()

    # support for arguments type: list of tuples (input, ex_output)
    if len(args) == 1:
        args = args[0]
    elif len(args) == 2:
        inputs, ex_outputs = args
        args = list(zip(inputs, ex_outputs))

    population = [Network(net.model_shape) for _ in range(params.POPULATION_SIZE)]

    for iter in range(params.MAX_GENERATIONS):
        semi_step = []

        for batch_ptr in range(0, len(args), params.BATCH_SIZE):
            inout = args[batch_ptr:batch_ptr+params.BATCH_SIZE]

            def fitness(network):
                costs = []
                for input, ex_output in inout:
                    output = network(input)
                    costs += [np.float_power(output - np.matrix(ex_output), 2)]
                return np.mean(costs)

            # selection

            # sorting by cost

            population.sort(key=lambda a: fitness(a))
            if kwargs.get("test_network_simple"):
                steps += [fitness(population[0])]  # smallest cost for statistics
            else:
                semi_step += [fitness(population[0])]  # smallest cost for statistics


            # crossover
            children = [copy.deepcopy(population[0])
                        for _ in range(params.POPULATION_SIZE - 2 // 3)]
            children += [copy.deepcopy(population[1])
                         for _ in range((params.POPULATION_SIZE - 2) // 3)]
            children += [network.average([population[0], population[1]])
                         for _ in range((params.POPULATION_SIZE - 2) // 3)]

            # mutation
            for individual in children:
                for layer in individual.layers:
                    w, b, _, _ = layer

                    for i in range(len(w)):
                        for j in range(len(w[i])):
                            if random.random() < params.MUTATION_CHANCE:
                                w[i][j] += random.uniform(-params.MUTATION_RATE, params.MUTATION_RATE)
                    for i in range(len(b)):
                        for j in range(len(b[i])):
                            if random.random() < params.MUTATION_CHANCE:
                                b[i][j] += random.uniform(-params.MUTATION_RATE, params.MUTATION_RATE)

            population = population[:2] + children[:params.POPULATION_SIZE - 2]


        if not kwargs.get("test_network_simple"):
            steps += [np.mean(semi_step)]

        if params.MAX_GENERATIONS > 200:
            progress_bar(iter, params.MAX_GENERATIONS)

        net.layers = population[0].layers

    return steps
