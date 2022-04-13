import neurolab as nl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Functions import get_function
from Genetic_Algorithms import *
from Gradient_Algorithm import *
from Network import network


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


def test_neuron_libs(lib="neurolab"):
    if lib == "neurolab":
        # __Model z losowymi danymy__
        # input = np.random.uniform(-0.5, 0.5, (10, 2))
        # print(input)
        # target = (input[:, 0] + input[:, 1]).reshape(10, 1)
        # net = nl.net.newff([[-0.5, 0.5], [-0.5, 0.5]], [5, 1])
        # err = net.train(input, target, show=15)
        # print("0.2 + 0.1 ?= ", end="")
        # print(net.sim([[0.2, 0.1]]))  # 0.2 + 0.1 array([[ 0.28757596]])

        # __Odwzorowanie tego twojego wyżej__
        input = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1]).reshape(4, 3)
        print(input)
        target = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]).reshape(4, 4)
        net = nl.net.newff([[0, 1], [0, 1], [0, 1]], [5, 5, 4])
        err = net.train(input, target)

        print(net.sim([[0, 0, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1]]))
        # oczekiwane wyniki:
        # 1 1 1 1
        # 1 0 1 1
        # 1 1 0 0
        # 0 0 0 0
    elif lib == "pytorch":
        # https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
        class Net(nn.Module):

            def __init__(self):
                super(Net, self).__init__()
                # 1 input image channel, 6 output channels, 5x5 square convolution
                # kernel
                self.conv1 = nn.Conv2d(1, 6, 5)
                self.conv2 = nn.Conv2d(6, 16, 5)
                # an affine operation: y = Wx + b
                self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

            def forward(self, x):
                # Max pooling over a (2, 2) window
                x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
                # If the size is a square, you can specify with a single number
                x = F.max_pool2d(F.relu(self.conv2(x)), 2)
                x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        # network
        net = Net()
        input = torch.randn(1, 1, 32, 32)
        print("input:")
        print(input)
        out = net(input)
        print("output:")
        print(out)

        # loss function
        output = net(input)
        target = torch.randn(10)  # a dummy target, for example
        target = target.view(1, -1)  # make it the same shape as output
        criterion = nn.MSELoss()

        loss = criterion(output, target)

        net.zero_grad()
        loss.backward()

        learning_rate = 0.01
        for f in net.parameters():
            f.data.sub_(f.grad.data * learning_rate)

        out = net(input)
        print("output:")
        print(out)

        # With square kernels and equal stride
        m = nn.Conv2d(16, 33, 3)
        input = torch.randn(20, 16, 50, 100)
        print("input test\n", input)
        output = m(input)
        #print("test \n",output)
