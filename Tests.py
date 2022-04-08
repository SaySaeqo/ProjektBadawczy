from Functions import get_function
from Genetic_Algorithms import *
from Gradient_Algorithm import *
from Network import network

import numpy as np
import neurolab as nl
from scipy.optimize import fmin_bfgs
from Functions import sigmoid


def get_algorithm(index):
    '''

    :param index: indeks algorytmu
    :return: krotka (algorytm, nazwa algorytmu)
    '''
    if index == 0:
        return 'gradient', gradient_algorithm
    elif index == 1:
        return 'genetic_func_mean_random', genetic_func_mean_random
    elif index == 2:
        return 'genetic_func_mean_change', genetic_func_mean_change
    elif index == 3:
        return 'genetic_func_mean_gradient_change_litle', genetic_func_mean_gradient_change_litle
    elif index == 4:
        return 0


def calc_error(result, solutions):
    '''

    :param result:
    :param solutions: tabica krotek rozwiązań
    :return:
    '''
    min_error = -1
    for i in range(len(solutions)):
        error = 0
        for j in range(len(result)):
            error += abs(result[j] - solutions[i][j])
        if min_error == -1 or error < min_error:
            min_error = error
    return min_error


def test_algorithms():
    stats_per_algorithm = []
    # [ (nazwa,   [(funkcja,blad),(funkcja,blad)   ) ,...   ]
    # dla każdego algorytmu
    for i in range(ALGORITHMS_NUMBER):
        # testujemy każdą funkcję
        algoritm_name, algorithm = get_algorithm(i)
        func_errors = []
        for j in range(FUNCTIONS_NUMBER):
            func = get_function(j + 1)
            result = algorithm(func.function, func.arg_num, func.domain, func.min_max, PROBE_NUMBER)
            error = calc_error(result, func.solutions)
            func_errors.append((j + 1, error))
        alg_errors = (algoritm_name, func_errors)
        stats_per_algorithm.append(alg_errors)

    # wypisanie wyników
    for i in range(len(stats_per_algorithm)):
        print(stats_per_algorithm[i][0])
        for j in range(len(stats_per_algorithm[i][1])):
            print('  ', stats_per_algorithm[i][1][j])


# testowanie gradientu, learn_rate hardcodowany (stara wersja dla 4 funkcji)
def test_gradient_learnrate():
    for i in range(1, 5):
        lowest = math.inf
        it = math.inf
        ar = 0
        for _ in range(100):
            func, start = Functions.get_function(i)
            lr = 0.00003
            if i == 3:
                lr = 150.0 / 300000.0
            args = gradient_func(func, start,
                                 learn_rate=lr, tol=0.0000001, max_iter=10_000)
            fx = func(args)
            if lowest > fx:
                lowest = fx
                ar = args
        print(i, ar, lowest)


def test_neuron():
    N = 1000
    nrn_nmb = [3, 5, 5, 4]
    net = network(nrn_nmb, net_gradient)

    # print (net)
    for i in range(N):
        net.process([0, 0, 0])
        net.correct([1, 1, 1, 1])
        net.process([0, 1, 0])
        net.correct([1, 0, 1, 1])
        net.process([0, 1, 1])
        net.correct([1, 1, 0, 0])
        net.process([1, 1, 1])
        net.correct([0, 0, 0, 0])
        print(f"\r{i}/{N}", end="")
    # print(net)
    #                          oczekiwane wyniki:
    print(net.process([0, 0, 0]))  # 1 1 1 1
    print(net.process([0, 1, 0]))  # 1 0 1 1
    print(net.process([0, 1, 1]))  # 1 1 0 0
    print(net.process([1, 1, 1]))  # 0 0 0 0

def test_neuron_libs():
    # Define Sequential model with 3 layers
    input = np.random.uniform(-0.5, 0.5, (10, 2))
    print(input)
    target = (input[:, 0] + input[:, 1]).reshape(10, 1)
    net = nl.net.newff([[-0.5, 0.5], [-0.5, 0.5]], [5, 1])
    err = net.train(input, target, show=15)
    print("0.2 + 0.1 ?= ", end="")
    print(net.sim([[0.2, 0.1]]))  # 0.2 + 0.1 array([[ 0.28757596]]) ```