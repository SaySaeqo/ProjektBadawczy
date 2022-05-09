import neurolab as nl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

from utils import Function
from genetic import *
from gradient import *
from network import Network


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
        j = 0
        for func in Function.get_all():
            result = algorithm(func.function, func.arg_num, func.domain, func.min_max, PROBE_NUMBER)
            error = calc_error(result, func.solutions)
            func_errors.append((j + 1, error))
            j += 1
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
        functions = Function.get_all()
        for _ in range(100):
            func = functions[i]
            start = func.random_data()
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
    model = [3,5,5, 4]
    net = Network(model)

    print (net)
    inputs = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    targets = [[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    steps = net.correct(inputs, targets)
    #print(f"\r{i}/{N}", end="")
    print("----------\nPO TRENINGU\n----------")
    print(net)
    #                          oczekiwane wyniki:
    print(net.process([0.0, 0.0, 0.0]))  # 1 1 1 1
    print(net.process([0.0, 1.0, 0.0]))  # 1 0 1 1
    print(net.process([0.0, 1.0, 1.0]))  # 1 1 0 0
    print(net.process([1.0, 1.0, 1.0]))  # 0 0 0 0
    print("1 1 1 1")
    print("1 0 1 1")
    print("1 1 0 0")
    print("0 0 0 0")

    plt.figure()
    plt.plot(steps[0:-4:4])
    plt.plot(steps[1:-3:4])
    plt.plot(steps[2:-2:4])
    plt.plot(steps[3:-1:4])
    plt.legend(["000->1111", "010->1011", "011->1100", "111->0000"])
    plt.xlabel("iterations")
    plt.ylabel("cost")
    plt.grid(linestyle='--')
    plt.show()


def test_neuron_libs(lib="neurolab"):
    if lib == "neurolab":
        print("NEUROLAB LIBRARY\n")

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
        print("PYTORCH LIBRARY\n")
        # https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
        class Net(nn.Module):

            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(3, 5)
                self.fc2 = nn.Linear(5, 5)
                self.fc3 = nn.Linear(5, 4)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        # network
        net = Net()
        input = torch.Tensor([[0, 0, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1]])
        print("input:")
        print(input)
        out = net(input)
        print("output (random weights):")
        print(out)

        # loss function
        criterion = nn.MSELoss()

        output = out
        target = torch.Tensor([[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]])
        # target = target.view(1, -1)  # make it the same shape as output

        # create your optimizer
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        N = 2**15
        for i in range(N):
            # training
            optimizer.zero_grad()  # zero the gradient buffers
            output = net(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()  # Does the update
            print(f"\r{i}/{N}", end="")

        out = net(input)
        print("output:")
        print(out)
        # oczekiwane wyniki:
        # 1 1 1 1
        # 1 0 1 1
        # 1 1 0 0
        # 0 0 0 0
