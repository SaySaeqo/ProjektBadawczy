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


def net_genetic(net: Network, inputs, targets, **kwargs):
    """
    Correct function for neural network. Uses **genetic algorithm** to upgrade network weights and biases.

    :param inputs: list of network's inputs
        (when single input for network is a list, then it is list of lists)
    :param expected_outputs: list of expected outputs for each network's inputs in previous argument
    :return: history for debugging and ploting
    """

    params = GeneticConst.instance()
    history = {
        "all_costs": [],
        "av_costs": [],
        "success_rate": []
    }

    train_data = list(zip(inputs, targets))

    population = [Network(net.model_shape) for _ in range(params.POPULATION_SIZE)]

    for gen in range(params.MAX_GENERATIONS):
        gen_costs = []

        # train
        for batch_ptr in range(0, len(train_data), params.BATCH_SIZE):
            inout = train_data[batch_ptr:batch_ptr + params.BATCH_SIZE]

            def fitness(network):
                costs = (np.sum(np.float_power(np.subtract(network(input), ex_output), 2))
                         for input, ex_output in inout)
                return sum(costs)/len(inout)

            # selection
            population.sort(key=lambda a: fitness(a))

            gen_costs += [fitness(population[0])]  # smallest cost for statistics

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
                    for i in range(len(layer["w"])):
                        for j in range(len(layer["w"][i])):
                            if random.random() < params.MUTATION_CHANCE:
                                layer["w"][i][j] += random.uniform(-params.MUTATION_RATE, params.MUTATION_RATE)
                    for i in range(len(layer["b"])):
                        for j in range(len(layer["b"][i])):
                            if random.random() < params.MUTATION_CHANCE:
                                layer["b"][i][j] += random.uniform(-params.MUTATION_RATE, params.MUTATION_RATE)

            population = population[:2] + children[:params.POPULATION_SIZE - 2]

        if kwargs.get("test_network_simple"):
            history["all_costs"] += gen_costs
        history["av_costs"] += [np.mean(gen_costs)]

        if kwargs.get("test_data"):
            success_rate = 0
            for data in kwargs["test_data"]:
                results = population[0](data[0])
                if all(abs(a - y) < 0.5 for a, y in zip(results, data[1])):
                    success_rate += 1
            success_rate /= len(kwargs["test_data"])
            history["success_rate"] += [success_rate]
            if success_rate > 0.9:
                break
            if len(history["success_rate"]) > 10 and \
                    success_rate <= sum(history["success_rate"][-11:-1]) / 10 and \
                    history["av_costs"][-1] >= sum(history["av_costs"][-11:-1]) / 10:
                break

        if params.SHOW_PROGRESS:
            progress_bar(gen, params.MAX_GENERATIONS)

        net.layers = population[0].layers

    return history


def net_genetic_wlt(net: Network, inputs, targets, **kwargs):
    """
    Correct function for neural network. Uses  **winning lotery ticket strategy**
    (genetic algorithm) to remove neurons from nets and get best subnet for the problem.

    :param inputs: list of network's inputs
        (when single input for network is a list, then it is list of lists)
    :param expected_outputs: list of expected outputs for each network's inputs in previous argument
    :return: history for debugging and ploting
    """
    params = GeneticConst.instance()
    history = {
        "all_costs": [],
        "av_costs": [],
        "success_rate": []
    }

    train_data = list(zip(inputs, targets))

    population = [network.WLTNetwork(net.model_shape) for _ in range(params.POPULATION_SIZE)]

    for gen in range(params.MAX_GENERATIONS):
        gen_costs = []

        for batch_ptr in range(0, len(train_data), params.BATCH_SIZE):
            inout = train_data[batch_ptr:batch_ptr + params.BATCH_SIZE]

            def fitness(network):
                costs = (np.sum(np.float_power(np.subtract(network(input), ex_output), 2))
                         for input, ex_output in inout)
                return np.mean(costs)

            # selection
            population.sort(key=lambda a: fitness(a))
            gen_costs += [fitness(population[0])]  # smallest cost for statistics

            # crossover
            children = []
            for _ in range(params.POPULATION_SIZE - 2):
                mather_genome = population[0].weights_genome
                tmp = []
                for x in mather_genome:
                    tmp += x.flatten().tolist()
                mather_genome = tmp
                father_genome = population[1].weights_genome
                tmp = []
                for x in father_genome:
                    tmp += x.flatten().tolist()
                father_genome = tmp

                nb_gens2take = random.randint(0, len(mather_genome))
                gens2take = random.sample(list(range(0, len(mather_genome))), nb_gens2take)
                child = network.WLTNetwork(net.model_shape)
                child_genome = []
                for i in gens2take:
                    child_genome += father_genome[len(child_genome):i]
                    child_genome += [mather_genome[i]]
                child_genome += father_genome[len(child_genome):len(father_genome)]

                i = 0
                for a in range(len(child.weights_genome)):
                    for b in range(child.weights_genome[a].shape[0]):
                        for c in range(child.weights_genome[a].shape[1]):
                            child.weights_genome[a][b][c] = child_genome[i]
                            i += 1
                children += [child]

            # mutation
            for child in children:
                for a in range(len(child.weights_genome)):
                    for b in range(child.weights_genome[a].shape[0]):
                        for c in range(child.weights_genome[a].shape[1]):
                            if random.random() < params.MUTATION_CHANCE:
                                child.weights_genome[a][b][c] = int(not child.weights_genome[a][b][c])
                # if random.random() < params.MUTATION_CHANCE:
                #     # nb_gens2reverse = random.randrange(
                #     #     len(individual.activation_genome) - individual.model_shape[-1])
                #
                #     # MUTATION_TYPE = "soft"
                #
                #     # if MUTATION_TYPE == "soft":
                #     #     if individual.activation_genome[nb_gens2reverse] == 0:
                #     #         individual.activation_genome[nb_gens2reverse] = 1
                #     #     else:
                #     #         individual.activation_genome[nb_gens2reverse] = 0
                #     # if MUTATION_TYPE == "hard":
                #     #     gens2reverse = random.sample(
                #     #         list(range(
                #     #             (len(individual.activation_genome) - individual.model_shape[-1])
                #     #         )),
                #     #         nb_gens2reverse)
                #     #     for i in gens2reverse:
                #     #         if individual.activation_genome[i] == 0:
                #     #             individual.activation_genome[i] = 1
                #     #         else:
                #     #             individual.activation_genome[i] = 0

            population = population[:2] + children[:params.POPULATION_SIZE - 2]

        if kwargs.get("test_network_simple"):
            history["all_costs"] += gen_costs
        history["av_costs"] += [np.mean(gen_costs)]

        if kwargs.get("test_data"):
            success_rate = 0
            for data in kwargs["test_data"]:
                results = population[0](data[0])
                if all(abs(a - y) < 0.5 for a, y in zip(results, data[1])):
                    success_rate += 1
            success_rate /= len(kwargs["test_data"])
            history["success_rate"] += [success_rate]
            if success_rate > 0.9:
                break
            if len(history["success_rate"]) > 10 and \
                    success_rate <= sum(history["success_rate"][-11:-1]) / 10 and \
                    history["av_costs"][-1] >= sum(history["av_costs"][-11:-1]) / 10:
                break

        if params.SHOW_PROGRESS:
            progress_bar(gen, params.MAX_GENERATIONS)

        net.layers = population[0].layers

    return history
