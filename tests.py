import csv
import random
import time

import neurolab as nl
import numpy as np
import texttable as texttable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

from utils import Function, progress_bar, sigmoid
from genetic import *
from gradient import *
from network import Network, WLTNetwork


def get_algorithm(index):
    """

    :param index: indeks algorytmu
    :return: krotka (algorytm, nazwa algorytmu)
    """
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
    """

    :param result:
    :param solutions: tabica krotek rozwiązań
    :return:
    """
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


def test_network_simple(correct_func):
    model = [3, 5, 5, 4]
    net = Network(model, correct_func=correct_func)

    print(net)
    inputs = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    targets = [[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    history = net.train(inputs, targets, test_network_simple=True)
    # print(f"\r{i}/{N}", end="")
    print("-*-*-*-*-*-*-*-*-*-\nPO TRENINGU\n-*-*-*-*-*-*-*-*-*-")
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
    plt.plot(history["all_costs"][0:-4:4])
    plt.plot(history["all_costs"][1:-3:4])
    plt.plot(history["all_costs"][2:-2:4])
    plt.plot(history["all_costs"][3:-1:4])
    plt.title(str(correct_func))
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
        N = 2 ** 15
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


def get_db(filename, name2tuple):
    with open(filename) as file:
        csvreader = csv.reader(file)

        table = []
        for row in csvreader:
            table.append(row)

        random.shuffle(table)

        inputs = []
        targets = []
        for row in table:
            targets += [name2tuple(row[-1])]
            inputs += [[float(elem) for elem in row[:-1]]]

        # standardization / normalization
        for feature_idx in range(len(inputs[0])):
            min_val = min(input[feature_idx] for input in inputs)
            for input in inputs:
                input[feature_idx] -= min_val
            max_val = max(input[feature_idx] for input in inputs)
            for input in inputs:
                input[feature_idx] /= max_val

    return list(zip(inputs, targets))


def test_network(database, net_model, train_func, test_data_length=3):
    # preparing data
    test_data = database[-test_data_length:]
    train_data = database[:-test_data_length]
    inputs = [data[0] for data in train_data]
    targets = [data[1] for data in train_data]
    net = Network(net_model, train_func)

    # training
    start = time.perf_counter()
    history = net.train(inputs, targets, test_data=test_data)
    time_passed = time.perf_counter() - start

    return history, time_passed


# def test_network(database, net_model, train_func, test_data_length=3, nb_tests=1):
#     av_costs = []
#     av_success_rate = []
#     av_time = []
#     for _ in range(nb_tests):
#         history, time_passed = test_network_once(database, net_model, train_func, test_data_length)
#         random.shuffle(database)
#         av_costs += [history["av_costs"]]
#         av_success_rate += [history["success_rate"]]
#         av_time += [time_passed]
#     av_costs = [sum(epoch) / len(epoch) for epoch in zip(*av_costs)]
#     av_success_rate = [sum(epoch) / len(epoch) for epoch in zip(*av_success_rate)]
#     av_time = sum(av_time) / len(av_time)
#
#     return {"av_costs": av_costs, "av_success_rate": av_success_rate, "av_time": av_time}


def plot_network_comparison(net_data, net_model, test_data_length=3):
    # start plotting
    fig, ax = plt.subplots(2, 1)
    plt.title("Neural network training method comparison")
    fig.tight_layout(pad=1.8)
    major_ticks = np.arange(1.15, step=0.1)
    minor_ticks = np.arange(1.1, step=0.05)

    #############          GRADIENT          #############
    history, time_passed = test_network(net_data, net_model, [net_gradient, net_genetic], test_data_length)
    print("Gradient time:", int(time_passed // 60), "min", time_passed % 60, "sec")

    # steps for gradient
    ax[0].plot(history["av_costs"], color="blue", label="cost")
    ax[0].plot(history["success_rate"], color="yellow", label="success_rate")
    for p in history["change_points"]:
        ax[0].vlines(p, 0.0, 1.0, alpha=0.5, color="red")
    ax[0].set(title="Av. Learn Error- gradient", xlabel="epoch", ylabel="value", ylim=[0.0, 1.0])
    # grid
    ax[0].grid(linestyle='--', which="major", alpha=0.5)
    ax[0].set_yticks(minor_ticks, minor=True)
    ax[0].set_yticks(major_ticks)
    # trend line
    domain = list(range(len(history["av_costs"])))
    z = numpy.polyfit(domain, history["av_costs"], 1)
    p = numpy.poly1d(z)
    ax[0].plot(domain, p(domain), "r--")
    ax[0].legend(["cost", "success rate", "training trend line"])

    #############          GENETIC          #############
    history, time_passed = test_network(net_data, net_model, [net_genetic, net_gradient], test_data_length)
    print("Genetic time:", int(time_passed // 60), "min", time_passed % 60, "sec")

    # steps for genetic
    ax[1].plot(history["av_costs"], color="blue", label="cost")
    ax[1].plot(history["success_rate"], color="yellow", label="success_rate")
    for p in history["change_points"]:
        ax[1].vlines(p, 0.0, 1.0, alpha=0.5, color="red")
    ax[1].set(title="Av. Learn Error- genetic", xlabel="generation", ylabel="value", ylim=[0.0, 1.0])
    # grid
    ax[1].grid(linestyle='--', which="major", alpha=0.5)
    ax[1].set_yticks(minor_ticks, minor=True)
    ax[1].set_yticks(major_ticks)
    # trend line
    domain = list(range(len(history["av_costs"])))
    z = numpy.polyfit(domain, history["av_costs"], 1)
    p = numpy.poly1d(z)
    ax[1].plot(domain, p(domain), "r--")
    ax[1].legend(["cost", "success rate", "training trend line"])


def plot_network_mix(net_data, net_model, test_data_length=3):
    # start plotting
    fig, ax = plt.subplots()
    plt.title("Neural network training method comparison")
    fig.tight_layout(pad=1.8)
    major_ticks = np.arange(1.15, step=0.1)
    minor_ticks = np.arange(1.1, step=0.05)

    #############          MIX          #############
    history, time_passed = test_network(net_data, net_model, [net_gradient, net_genetic], test_data_length)
    print("Mix time:", int(time_passed // 60), "min", time_passed % 60, "sec")

    # steps for genetic
    ax.plot(history["av_costs"], color="blue", label="cost")
    ax.plot(history["success_rate"], color="yellow", label="success_rate")
    for p in history["change_points"]:
        ax.vlines(p, 0.0, 1.0, alpha=0.5, color="red")
    ax.set(title="Av. Learn Error- mix", xlabel="iteration", ylabel="value", ylim=[0.0, 1.0])
    # grid
    ax.grid(linestyle='--', which="major", alpha=0.5)
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    # trend line
    domain = list(range(len(history["av_costs"])))
    z = numpy.polyfit(domain, history["av_costs"], 1)
    p = numpy.poly1d(z)
    ax.plot(domain, p(domain), "r--")
    ax.legend(["cost", "success rate", "training trend line"])


def test_iris(nb_tests=1):
    """
    That function just gather parameters I choose good in way of trail and fails process
    """
    print("Testing Iris DB...")

    def name2tuple(name):
        if name == "Iris-setosa":
            return (0, 0)
        elif name == "Iris-versicolor":
            return (0, 1)
        elif name == "Iris-virginica":
            return (1, 1)

    net_data = get_db("iris.data", name2tuple)
    net_model = [4, 5, 5, 2]
    test_data_length = len(net_data) // 20 * 3  # 15%

    # parameters for genetic
    gen_params = GeneticConst.instance()
    gen_params.MAX_GENERATIONS = 100
    gen_params.POPULATION_SIZE = 8
    gen_params.MUTATION_CHANCE = 0.9
    gen_params.MUTATION_RATE = 0.2
    gen_params.BATCH_SIZE = 10
    # parameters for gradient
    grad_params = GradientConst.instance()
    grad_params.MAX_EPOCHS = 100
    grad_params.BATCH_SIZE = 10

    # plot results
    if nb_tests > 1:
        plot_network_overall_performance_comparison(nb_tests, net_data, net_model, test_data_length)
        plt.savefig("last_iris_performance.png")
    else:
        plot_network_comparison(net_data, net_model, test_data_length)
        plt.savefig("last_iris.png")
    table = texttable.Texttable()
    table.add_row(["Gradient params:", "Genetic params:"])
    table.add_row([repr(grad_params), repr(gen_params)])
    print(table.draw())


def test_raisin(nb_tests=1):
    """
    That function just gather parameters I choose good in way of trail and fails process
    """
    print("Testing Raisin DB...")

    def name2tuple(name):
        if name == "Kecimen":
            return [0]
        elif name == "Besni":
            return [1]

    net_data = get_db("raisin.data", name2tuple)
    net_model = [7, 5, 5, 1]
    test_data_length = len(net_data) // 20 * 3  # 15%

    # parameters for genetic
    gen_params = GeneticConst.instance()
    gen_params.MAX_GENERATIONS = 100
    gen_params.POPULATION_SIZE = 8
    gen_params.MUTATION_CHANCE = 0.9
    gen_params.MUTATION_RATE = 0.2
    gen_params.BATCH_SIZE = 15
    # parameters for gradient
    grad_params = GradientConst.instance()
    grad_params.MAX_EPOCHS = 100
    grad_params.BATCH_SIZE = 3

    # plot results
    if nb_tests > 1:
        plot_network_overall_performance_comparison(nb_tests, net_data, net_model, test_data_length)
        plt.savefig("last_raisin_performance.png")
    else:
        plot_network_comparison(net_data, net_model, test_data_length)
        plt.savefig("last_raisin.png")
    table = texttable.Texttable()
    table.add_row(["Gradient params:", "Genetic params:"])
    table.add_row([repr(grad_params), repr(gen_params)])
    print(table.draw())


def test_beans(nb_tests=1):
    """
    That function just gather parameters I choose good in way of trail and fails process
    """
    print("Testing Beans DB...")

    def name2tuple(name):
        if name == "SEKER":
            return (0, 0, 0)
        elif name == "BARBUNYA":
            return (0, 0, 1)
        elif name == "BOMBAY":
            return (0, 1, 0)
        elif name == "CALI":
            return (0, 1, 1)
        elif name == "DERMASON":
            return (1, 0, 0)
        elif name == "HOROZ":
            return (1, 0, 1)
        elif name == "SIRA":
            return (1, 1, 0)

    net_data = get_db("dry_bean.data", name2tuple)
    net_model = [16, 10, 10, 3]
    test_data_length = len(net_data) // 20 * 3  # 15%

    # parameters for genetic
    gen_params = GeneticConst.instance()
    gen_params.MAX_GENERATIONS = 100
    gen_params.POPULATION_SIZE = 8
    gen_params.MUTATION_CHANCE = 0.8
    gen_params.MUTATION_RATE = 0.2
    gen_params.BATCH_SIZE = 40
    gen_params.SHOW_PROGRESS = nb_tests <= 1
    # parameters for gradient
    grad_params = GradientConst.instance()
    grad_params.MAX_EPOCHS = 100
    grad_params.BATCH_SIZE = 10
    grad_params.SHOW_PROGRESS = nb_tests <= 1

    # plot results
    if nb_tests > 1:
        plot_network_overall_performance_comparison(nb_tests, net_data, net_model, test_data_length)
        plt.savefig("last_beans_performance.png")
    else:
        plot_network_comparison(net_data, net_model, test_data_length)
        plt.savefig("last_beans.png")
    table = texttable.Texttable()
    table.add_row(["Gradient params:", "Genetic params:"])
    table.add_row([repr(grad_params), repr(gen_params)])
    print(table.draw())


def test_network_overall_performance(nb_tests, database, net_model, train_func, test_data_length=3):
    time_over_tests = []
    sr_over_tests = []
    cost_over_tests = []
    for i in range(nb_tests):
        random.shuffle(database)
        history, time_passed = test_network(database, net_model, train_func, test_data_length)
        time_over_tests += [time_passed]
        sr_over_tests += [history["success_rate"][-1]]
        cost_over_tests += [history["av_costs"][-1]]
        progress_bar(i, nb_tests)
    return cost_over_tests, sr_over_tests, time_over_tests


def plot_network_overall_performance_comparison(nb_tests, net_data, net_model, test_data_length=3):
    # start plotting
    fig, ax = plt.subplots(2, 2)
    plt.title("Neural network overall performance comparison")
    fig.tight_layout(pad=1.8)
    major_ticks = np.arange(1.15, step=0.1)
    minor_ticks = np.arange(1.1, step=0.05)

    #############          GRADIENT          #############
    costs, success_rates, times = test_network_overall_performance(
        nb_tests, net_data, net_model, net_gradient, test_data_length)

    # steps for gradient
    ax[0][0].plot(costs, color="blue", label="cost")
    ax[0][0].plot(success_rates, color="yellow", label="success_rate")
    ax[0][0].set(title="Av. Learn Error- gradient", xlabel="attempt", ylabel="value", ylim=[0.0, 1.0])
    # grid
    ax[0][0].grid(linestyle='--', which="major", alpha=0.5)
    ax[0][0].set_yticks(minor_ticks, minor=True)
    ax[0][0].set_yticks(major_ticks)
    ax[0][0].legend(["final cost", "final success rate"])
    # times
    ax[0][1].plot(times)
    ax[0][1].set(title="Time needed to train- gradient", xlabel="attempt", ylabel="seconds")

    #############          GENETIC          #############
    costs, success_rates, times = test_network_overall_performance(
        nb_tests, net_data, net_model, net_genetic, test_data_length)

    # steps for gradient
    ax[1][0].plot(costs, color="blue", label="cost")
    ax[1][0].plot(success_rates, color="yellow", label="success_rate")
    ax[1][0].set(title="Av. Learn Error- genetic", xlabel="attempt", ylabel="value", ylim=[0.0, 1.0])
    # grid
    ax[1][0].grid(linestyle='--', which="major", alpha=0.5)
    ax[1][0].set_yticks(minor_ticks, minor=True)
    ax[1][0].set_yticks(major_ticks)
    ax[1][0].legend(["final cost", "final success rate"])
    # times
    ax[1][1].plot(times)
    ax[1][1].set(title="Time needed to train- genetic", xlabel="attempt", ylabel="seconds")
