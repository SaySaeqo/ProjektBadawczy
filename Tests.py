import time

import neurolab as nl
import numpy as np
import csv
import matplotlib.pyplot as plt
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim

from Functions import get_function
from Genetic_Algorithms import *
from Gradient_Algorithm import *
from Mix_Network import *
import collections


def Compare_lists(a, b):
    if len(a) == len(b):
        for i in range(len(a)):
            if a[i] != b[i]:
                return False
        return True
    else:
        return False


def ParseOutpus(list):
    for i in range(len(list)):
        list[i] = round(list[i])
    return list


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
    '''
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
    '''

def PlotIris(option):
    ROWS = 150
    def name_to_tuple_iris(name):
        if name == "Iris-setosa":
            return [1, 0, 0]
        elif name == "Iris-versicolor":
            return [0, 1, 0]
        elif name == "Iris-virginica":
            return [0, 0, 1]

    #stworzenie bazy danych
    file = open("iris.data")
    csvreader = csv.reader(file)

    table = []
    for row in csvreader:
        table.append(row)

    random.shuffle(table)
    net_data = [None] * ROWS
    input = [None] * ROWS
    expected = [None] * ROWS
    for i in range(ROWS):
        row = table[i]
        name = row[4]
        row = row[:4]
        for j in range(4):
            row[j] = float(row[j])/7

        input[i] = row
        expected[i] = name_to_tuple_iris(name)
        net_data[i] = (row, expected[i])
        # print (row,expected)

    if option == "iris_single_try":
        LOOPS=20
        net = geneticNetwork([4, 3, 3, 3], 10, 15)
        net_normal = network([4, 3, 3, 3], net_gradient,1, learnBase = 8 ,fractionLearnRate=3/4, learnSuppresion = 200)
        error_genetic = []
        error_normal = []
        #tutaj operujemy na input i expected
        net.start_learning()
        net_normal.start_learning()
        for j in range(LOOPS):
            for i in range(ROWS):
                net.process(net_data[i][0])
                net.correct(net_data[i][1])
                net_normal.process(net_data[i][0])
                net_normal.correct(net_data[i][1])
                error_normal.append(net_normal.total_error(input,expected))
                error_genetic.append(net.total_error(input, expected))
            print(j)
        net_normal.stop_learning()
        net.stop_learning()

        domain=range(1,ROWS*LOOPS+1)

        plot1 = plt.subplot2grid((2,1), (0, 0))
        plot1.plot(domain, error_normal, label="price")
        plot1.set_xlabel("iteration")
        plot1.set_ylabel("total error")
        plot1.set_title("Learnig curve - gradient")
        #linia trendu \/
        z = numpy.polyfit(domain, error_normal, 1)
        p = numpy.poly1d(z)
        plot1.plot(domain, p(domain), "r--")

        #plt.plot(range(150), error_normal, label="oś x")
        plot2 = plt.subplot2grid((2, 1), (1, 0))
        plot2.plot(domain, error_genetic, label="price")
        plot2.set_xlabel("iteration")
        plot2.set_ylabel("total error")
        plot2.set_title("Learnig curve - genetic - batch size 5")
        # linia trendu \/
        z = numpy.polyfit(domain, error_genetic, 1)
        p = numpy.poly1d(z)
        plot2.plot(domain, p(domain), "r--")
        plt.show()

        for i in range(ROWS):
            print(net_normal.process(net_data[i][0]),net_data[i][1])
            print(net.process(net_data[i][0]), net_data[i][1])

    if option == "iris_genetic_learn_many_attempts":
        error_normal = []
        error_genetic = []
        domain=range(30)
        for j in domain:
            print(j)
            random.shuffle(net_data)
            # resetowanie sieci
            net = geneticNetwork([4, 3, 3, 3], 50)
            net_normal = network([4, 3, 3, 3], net_gradient)

            net.start_learning()
            net_normal.start_learning()
            for i in range(ROWS):
                net.process(net_data[i][0])
                net.correct(net_data[i][1])
                #net_normal.process(net_data[i][0])
                #net_normal.correct(net_data[i][1])

            net_normal.stop_learning()
            net.stop_learning()

            error_normal.append(net_normal.total_error(input, expected))
            error_genetic.append(net.total_error(input, expected))

        plot1 = plt.subplot2grid((2, 1), (0, 0))
        plot1.plot(domain, error_normal, label="price")
        plot1.set_xlabel("attempt")
        plot1.set_ylabel("total error")
        plot1.set_title("Learn Error- gradient")

        # plt.plot(range(150), error_normal, label="oś x")
        plot2 = plt.subplot2grid((2, 1), (1, 0))
        plot2.plot(domain, error_genetic, label="price")
        plot2.set_xlabel("attempt")
        plot2.set_ylabel("total error")
        plot2.set_title("Learnt Net Error - genetic")

        plt.show()
    if option == "iris_genetic_time_comparison":
        # a potem wielu sieci naraz
        # uczymy obie sieci od nowa 100 razy
        time_genetic = []
        time_gradient = []
        domain = range(10)
        for j in domain:
            print(j)
            random.shuffle(net_data)
            # resetowanie sieci
            net = geneticNetwork([4, 3, 3, 3], 50,5)
            net_normal = network([4, 3, 3, 3], net_gradient,5)

            start = time.time()
            net.start_learning()
            for i in range(150):
                net.process(net_data[i][0])
                net.correct(net_data[i][1])
            net.stop_learning()

            end = time.time()
            time_genetic.append(end-start)

            start = time.time()
            net_normal.start_learning()
            for i in range(150):
                net_normal.process(net_data[i][0])
                net_normal.correct(net_data[i][1])
            net_normal.stop_learning()

            end = time.time()
            time_gradient.append(end - start)

        plot1 = plt.subplot2grid((2, 1), (0, 0))
        plot1.plot(domain, time_gradient)
        plot1.set_xlabel("attempt")
        plot1.set_ylabel("time needed to learn[s]")
        plot1.set_title("Time spent on learning- gradient")

        plot2 = plt.subplot2grid((2, 1), (1, 0))
        plot2.plot(domain, time_genetic)
        plot2.set_xlabel("attempt")
        plot2.set_ylabel("time needed to learn[s]")
        plot2.set_title("Time spent on learning- genetic")

        plt.show()

    if option == "test_gradient":
        LOOPS = 10
        net_normal = network([4, 3, 3, 3], net_gradient,1, learnBase = 8 ,fractionLearnRate=3/4, learnSuppresion = 200)
        error= []
        # tutaj operujemy na input i expected
        net_normal.start_learning()
        for j in range(LOOPS):
            for i in range(ROWS):
                net_normal.process(net_data[i][0])
                net_normal.correct(net_data[i][1])
                error.append(net_normal.total_error(input, expected))
                print(j, i)
        net_normal.stop_learning()

        domain = range(1, ROWS * LOOPS + 1)

        # plt.plot(range(150), error_normal, label="oś x")
        plt.plot((1, 1), (1, 0))
        plt.plot(domain, error, label="price")
        plt.xlabel("iteration")
        plt.ylabel("total error")
        plt.title("Learnig curve - genetic - batch size 5")
        # linia trendu \/
        z = numpy.polyfit(domain, error, 1)
        p = numpy.poly1d(z)
        plt.plot(domain, p(domain), "r--")
        plt.show()

        for i in range(ROWS):
            print(net_normal.process(net_data[i][0]), net_data[i][1])

    if option == "test_genetic":
        LOOPS = 6
        net = geneticNetwork([4, 3, 3, 3],net_gradient, 50, 5, learnBase = 1 ,fractionLearnRate=3/4, learnSuppresion = 50)
        error_genetic = []
        # tutaj operujemy na input i expected
        net.start_learning()
        for j in range(LOOPS):
            for i in range(ROWS):
                net.process(net_data[i][0])
                net.correct(net_data[i][1])
                error_genetic.append(net.total_error(input, expected))
                print(j, i)
        net.stop_learning()

        domain = range(1, ROWS * LOOPS + 1)

        # plt.plot(range(150), error_normal, label="oś x")
        plt.plot((1, 1), (1, 0))
        plt.plot(domain, error_genetic, label="price")
        plt.xlabel("iteration")
        plt.ylabel("total error")
        plt.title("Learnig curve - genetic - batch size 5")
        # linia trendu \/
        z = numpy.polyfit(domain, error_genetic, 1)
        p = numpy.poly1d(z)
        plt.plot(domain, p(domain), "r--")
        plt.show()

        for i in range(ROWS):
            print(net.process(net_data[i][0]), net_data[i][1])


    if option == "test_generalism":

        LOOPS = 10
        LEARNING_LOOPS = 10
        TEST_PROBES=10
        # tutaj operujemy na input i expected
        accuracy_gradient=[]
        accuracy_genetic=[]
        for j in range(LOOPS):
            random.shuffle(net_data)
            print(j)
            net = geneticNetwork([4, 3, 3, 3], 50, 5)
            net_normal = network([4, 3, 3, 3], net_gradient, 1)
            net.start_learning()
            net_normal.start_learning()
            for _ in range(LEARNING_LOOPS):
                for i in range(ROWS-TEST_PROBES):
                    net.process(net_data[i][0])
                    net.correct(net_data[i][1])
                    net_normal.process(net_data[i][0])
                    net_normal.correct(net_data[i][1])
            net_normal.stop_learning()
            net.stop_learning()

            genetic_hits=0
            gradient_hits=0
            for i in range(ROWS-TEST_PROBES,ROWS):
                otp=ParseOutpus(net_normal.process(net_data[i][0]))
                if Compare_lists(otp,net_data[i][1]):
                    gradient_hits+=1
                otp = ParseOutpus(net.process(net_data[i][0]))
                if Compare_lists(otp,net_data[i][1]):
                    genetic_hits+=1

            accuracy_gradient.append(gradient_hits/TEST_PROBES)
            accuracy_genetic.append(genetic_hits/TEST_PROBES)


        domain = range(1, LOOPS + 1)

        plot1 = plt.subplot2grid((2, 1), (0, 0))
        plot1.plot(domain, accuracy_gradient, label="price")
        plot1.set_xlabel("attempt")
        plot1.set_ylabel("accuracy")
        plot1.set_title("Gradient Accuracy learning loops =10")
        # linia trendu \/
        z = numpy.polyfit(domain, accuracy_gradient, 1)
        p = numpy.poly1d(z)
        plot1.plot(domain, p(domain), "r--")

        # plt.plot(range(150), error_normal, label="oś x")
        plot2 = plt.subplot2grid((2, 1), (1, 0))
        plot2.plot(domain, accuracy_genetic, label="price")
        plot2.set_xlabel("attempt")
        plot2.set_ylabel("accuracy")
        plot2.set_title("Genetic Accuracy TEST_PROBES=10")
        # linia trendu \/
        z = numpy.polyfit(domain, accuracy_genetic, 1)
        p = numpy.poly1d(z)
        plot2.plot(domain, p(domain), "r--")
        plt.show()

    if option == "test_mix":
        LOOPS = 15
        mix_net = mixNetwork([4, 3, 3, 3],net_gradient, 25,30,2,input,expected,50, learnBase = 4 ,fractionLearnRate=3/4, learnSuppresion = 400)
        error_genetic = []
        accuracy_mix = []
        domain = range(1, ROWS * LOOPS + 1,10)
        # tutaj operujemy na input i expected
        mix_net.start_learning()
        for j in range(LOOPS):
            for i in range(ROWS):
                mix_net.process(net_data[i][0])
                mix_net.correct(net_data[i][1])
                if (j*ROWS +i) in domain:
                    error_genetic.append(mix_net.total_error(input, expected))
                    gradient_hits = 0
                    for k in range(ROWS):
                        otp = ParseOutpus(mix_net.process(net_data[k][0]))
                        if Compare_lists(otp, net_data[k][1]):
                            gradient_hits += 1
                    accuracy_mix.append(gradient_hits)
                print(j, i)
        mix_net.stop_learning()

        # plt.plot(range(150), error_normal, label="oś x")
        plt.plot((1, 1), (1, 0))
        plt.plot(domain, error_genetic, label="price")
        plt.plot(domain, accuracy_mix, label="hits")
        plt.xlabel("iteration")
        plt.ylabel("total error")
        plt.title("Learnig curve - mix - batch size 5")
        # linia trendu \/
        z = numpy.polyfit(domain, error_genetic, 1)
        p = numpy.poly1d(z)
        plt.plot(domain, p(domain), "r--")
        plt.show()

        for i in range(ROWS):
            print(mix_net.process(net_data[i][0]), net_data[i][1])


def PlotBeans(option):
    ROWS = 13611 #nadpisuje, bo dla większej liczby to trwa wieki XD

    def name_to_tuple_beans(name):
        if name == "SEKER":
            return [1, 0, 0, 0, 0, 0, 0]
        elif name == "BARBUNYA":
            return [0, 1, 0, 0, 0, 0, 0]
        elif name == "BOMBAY":
            return [0, 0, 1, 0, 0, 0, 0]
        elif name == "CALI":
            return [0, 0, 0, 1, 0, 0, 0]
        elif name == "DERMASON":
            return [0, 0, 0, 0, 1, 0, 0]
        elif name == "HOROZ":
            return [0, 0, 0, 0, 0, 1, 0]
        elif name == "SIRA":
            return [0, 0, 0, 0, 0, 0, 1]

    file = open("DryBeanDataset/Dry_Bean_Dataset.arff")
    csvreader = csv.reader(file)

    table = []
    for row in csvreader:
        table.append(row)

    random.shuffle(table)
    net_data = [None] * ROWS
    input = [None] * ROWS
    expected = [None] * ROWS
    for i in range(ROWS):
        row = table[i]
        name = row[16]
        row = row[:16]
        for j in range(16):
            row[j] = float(row[j]) #konwersja

        #poprawianie do przedziału (0,1)
        row[0] = row[0] / 254616
        row[1] = row[1] / 1986
        row[2] = row[2] / 739
        row[3] = row[3] / 462
        row[4] = row[4] / 2.5
        row[6] = row[6] / 263261



        input[i] = row
        expected[i] = name_to_tuple_beans(name)
        net_data[i] = (row, expected[i])

        # print (row,expected)

    if option == "beans_single_try":
        batch_size=100
        LOOPS=20
        domain = range(0,ROWS*LOOPS,batch_size)
        net = geneticNetwork([16, 8, 8, 7], 50,14)
        net_normal = network([16, 8, 8, 7], net_gradient,batch_size)
        #error_genetic = [0]*ROWS
        error_genetic = []
        error_normal = []
        #tutaj operujemy na input i expected
        net.start_learning()
        net_normal.start_learning()
        for j in range(LOOPS):
            for i in range(ROWS):
                net.process(net_data[i][0])
                net.correct(net_data[i][1])
                net_normal.process(net_data[i][0])
                net_normal.correct(net_data[i][1])
                if(j*ROWS+i) in domain:
                    error_normal.append(net_normal.total_error(input,expected))
                    error_genetic.append(net.total_error(input, expected))
                print(j,i)
        net_normal.stop_learning()
        net.stop_learning()

        print("domain len",len(domain))
        print("error len",len(error_normal))
        plot1 = plt.subplot2grid((2,1), (0, 0))
        plot1.plot(domain, error_normal, label="price")
        plot1.set_xlabel("iteration")
        plot1.set_ylabel("total error")
        plot1.set_title("Learnig curve - gradient")
        #linia trendu \/
        z = numpy.polyfit(domain, error_normal, 1)
        p = numpy.poly1d(z)
        plot1.plot(domain, p(domain), "r--")


        #plt.plot(range(150), error_normal, label="oś x")
        plot2 = plt.subplot2grid((2, 1), (1, 0))
        plot2.plot(domain, error_genetic, label="price")
        plot2.set_xlabel("iteration")
        plot2.set_ylabel("total error")
        plot2.set_title("Learnig curve - genetic")
        # linia trendu \/
        z = numpy.polyfit(domain, error_genetic, 1)
        p = numpy.poly1d(z)
        plot2.plot(domain, p(domain), "r--")

        plt.show()

    elif option == "beans_genetic_learn_many_attempts":

        error_normal = []
        error_genetic = []
        domain=range(10)
        for j in domain:
            print(j)
            random.shuffle(net_data)
            #resetowanie sieci
            net = geneticNetwork([16, 12, 10, 8, 7], 50)
            net_normal = network([16, 12, 10, 8, 7], net_gradient)

            net.start_learning()
            net_normal.start_learning()
            for i in range(150):
                net.process(net_data[i][0])
                net.correct(net_data[i][1])
                #net_normal.process(net_data[i][0])
                #net_normal.correct(net_data[i][1])

            net_normal.stop_learning()
            net.stop_learning()

            error_normal.append(net_normal.total_error(input, expected))
            error_genetic.append(net.total_error(input, expected))

        plot1 = plt.subplot2grid((2, 1), (0, 0))
        plot1.plot(domain, error_normal, label="price")
        plot1.set_xlabel("attempt")
        plot1.set_ylabel("total error")
        plot1.set_title("Learn Error- gradient")

        # plt.plot(range(150), error_normal, label="oś x")
        plot2 = plt.subplot2grid((2, 1), (1, 0))
        plot2.plot(domain, error_genetic, label="price")
        plot2.set_xlabel("attempt")
        plot2.set_ylabel("total error")
        plot2.set_title("Learnt Net Error - genetic")

        plt.show()

    elif option == "test_gradient":
        ROWS = 1000
        batch_size = 1

        net_normal = network([16, 12, 10, 8, 7], net_gradient, batch_size)
        # tutaj operujemy na input i expected
        net_normal.start_learning()
        for i in range(ROWS):
            net_normal.process(net_data[i][0])
            net_normal.correct(net_data[i][1])
            print(i)
        net_normal.stop_learning()

        for i in range(ROWS):
            print(net_normal.process(net_data[i][0]),net_data[i][1])

    if option == "test_generalism":
        def Compare_lists(a,b):
            if len(a)==len(b):
                for i in range(len(a)):
                    if a[i] != b[i]:
                        return False
                return True
            else:
                return False

        def ParseOutpus(list):
            for i in range(len(list)):
                list[i]=round(list[i])
            return list

        ROWS = 100
        LOOPS = 2
        LEARNING_LOOPS = 35
        TEST_PROBES=100
        # tutaj operujemy na input i expected
        accuracy_gradient=[]
        accuracy_genetic=[]
        for j in range(LOOPS):
            random.shuffle(net_data)
            print(j)
            random.shuffle(net_data)
            net = geneticNetwork([16, 12, 10, 8, 7], 50,2)
            net_normal = network([16, 12, 10, 8, 7], net_gradient,5)
            net.start_learning()
            net_normal.start_learning()
            for _ in range(LEARNING_LOOPS):
                for i in range(ROWS-TEST_PROBES):
                    net.process(net_data[i][0])
                    net.correct(net_data[i][1])
                    net_normal.process(net_data[i][0])
                    net_normal.correct(net_data[i][1])
            net_normal.stop_learning()
            net.stop_learning()

            genetic_hits=0
            gradient_hits=0
            for i in range(ROWS-TEST_PROBES,ROWS):
                otp=ParseOutpus(net_normal.process(net_data[i][0]))
                if Compare_lists(otp,net_data[i][1]):
                    gradient_hits+=1
                otp = ParseOutpus(net.process(net_data[i][0]))
                if Compare_lists(otp,net_data[i][1]):
                    genetic_hits+=1

            accuracy_gradient.append(gradient_hits/TEST_PROBES)
            accuracy_genetic.append(genetic_hits/TEST_PROBES)


        domain = range(1, LOOPS + 1)

        plot1 = plt.subplot2grid((2, 1), (0, 0))
        plot1.plot(domain, accuracy_gradient, label="price")
        plot1.set_xlabel("attempt")
        plot1.set_ylabel("accuracy")
        plot1.set_title("Gradient Accuracy learning loops =30")
        # linia trendu \/
        z = numpy.polyfit(domain, accuracy_gradient, 1)
        p = numpy.poly1d(z)
        plot1.plot(domain, p(domain), "r--")

        # plt.plot(range(150), error_normal, label="oś x")
        plot2 = plt.subplot2grid((2, 1), (1, 0))
        plot2.plot(domain, accuracy_genetic, label="price")
        plot2.set_xlabel("attempt")
        plot2.set_ylabel("accuracy")
        plot2.set_title("Genetic Accuracy TEST_PROBES=10")
        # linia trendu \/
        z = numpy.polyfit(domain, accuracy_genetic, 1)
        p = numpy.poly1d(z)
        plot2.plot(domain, p(domain), "r--")
        plt.show()

def PlotRisin(option):
    ROWS = 900
    def name_to_tuple_Risin(name):
        if name == "Kecimen":
            return [1,  0]
        elif name == "Besni":
            return [0, 1]

    #stworzenie bazy danych
    file = open("Raisin_Dataset/Raisin_Dataset.arff")
    csvreader = csv.reader(file)

    table = []
    for row in csvreader:
        table.append(row)

    random.shuffle(table)
    net_data = [None] * ROWS
    input = [None] * ROWS
    expected = [None] * ROWS
    for i in range(ROWS):
        row = table[i]
        name = row[7]
        row = row[:7]
        for j in range(7):
            row[j] = float(row[j])  # konwersja

        # poprawianie do przedziału (0,1)
        row[0] = row[0] / 235047
        row[1] = row[1] / 1000
        row[2] = row[2] / 493
        row[4] = row[4] / 278217
        row[6] = row[6] / 2700

        input[i] = row
        expected[i] = name_to_tuple_Risin(name)
        net_data[i] = (row, expected[i])
        # print (row,expected)

    if option == "risin_single_try":
        LOOPS=6
        net = geneticNetwork([7, 4, 4, 2], 10, 5)
        net_normal = network([7, 4, 4, 2], net_gradient,10)
        error_genetic = []
        error_normal = []
        #tutaj operujemy na input i expected
        net.start_learning()
        net_normal.start_learning()
        for j in range(LOOPS):
            for i in range(ROWS):
                net.process(net_data[i][0])
                net.correct(net_data[i][1])
                net_normal.process(net_data[i][0])
                net_normal.correct(net_data[i][1])
                error_normal.append(net_normal.total_error(input,expected))
                error_genetic.append(net.total_error(input, expected))
                print(j,i)
        net_normal.stop_learning()
        net.stop_learning()

        domain=range(1,ROWS*LOOPS+1)

        plot1 = plt.subplot2grid((2,1), (0, 0))
        plot1.plot(domain, error_normal, label="price")
        plot1.set_xlabel("iteration")
        plot1.set_ylabel("total error")
        plot1.set_title("Learnig curve - gradient")
        #linia trendu \/
        z = numpy.polyfit(domain, error_normal, 1)
        p = numpy.poly1d(z)
        plot1.plot(domain, p(domain), "r--")

        #plt.plot(range(150), error_normal, label="oś x")
        plot2 = plt.subplot2grid((2, 1), (1, 0))
        plot2.plot(domain, error_genetic, label="price")
        plot2.set_xlabel("iteration")
        plot2.set_ylabel("total error")
        plot2.set_title("Learnig curve - genetic - batch size 5")
        # linia trendu \/
        z = numpy.polyfit(domain, error_genetic, 1)
        p = numpy.poly1d(z)
        plot2.plot(domain, p(domain), "r--")
        plt.show()

        for i in range(ROWS):
            print(net_normal.process(net_data[i][0]),net_data[i][1])
            print(net.process(net_data[i][0]), net_data[i][1])

    if option == "test_generalism":

        LOOPS = 2
        LEARNING_LOOPS = 6
        TEST_PROBES = 50
        # tutaj operujemy na input i expected
        accuracy_gradient = []
        accuracy_genetic = []

        for j in range(LOOPS):
            random.shuffle(net_data)
            net = geneticNetwork([7, 6, 5, 2],net_gradient, 20, 100)
            net_normal = network([7, 4, 4, 2], net_gradient, 10, learnBase = 1 ,fractionLearnRate=1, learnSuppresion = 1)
            net.start_learning()
            net_normal.start_learning()
            for k in range(LEARNING_LOOPS):
                for i in range(ROWS - TEST_PROBES):
                    print(j,k,i)
                    net.process(net_data[i][0])
                    net.correct(net_data[i][1])
                    net_normal.process(net_data[i][0])
                    net_normal.correct(net_data[i][1])
            net_normal.stop_learning()
            net.stop_learning()

            genetic_hits = 0
            gradient_hits = 0
            for i in range(ROWS - TEST_PROBES, ROWS):
                otp = ParseOutpus(net_normal.process(net_data[i][0]))
                if Compare_lists(otp, net_data[i][1]):
                    gradient_hits += 1
                otp = ParseOutpus(net.process(net_data[i][0]))
                if Compare_lists(otp, net_data[i][1]):
                    genetic_hits += 1

            accuracy_gradient.append(gradient_hits / TEST_PROBES)
            accuracy_genetic.append(genetic_hits / TEST_PROBES)

        domain = range(1, LOOPS + 1)

        plot1 = plt.subplot2grid((2,1), (0, 0))
        plot1.plot(domain, accuracy_gradient, label="price")
        plot1.set_xlabel("iteration")
        plot1.set_ylabel("hit %")
        plot1.set_title("Learnig curve - gradient")
        #linia trendu \/
        z = numpy.polyfit(domain, accuracy_gradient, 1)
        p = numpy.poly1d(z)
        plot1.plot(domain, p(domain), "r--")

        #plt.plot(range(150), error_normal, label="oś x")
        plot2 = plt.subplot2grid((2, 1), (1, 0))
        plot2.plot(domain, accuracy_genetic, label="price")
        plot2.set_xlabel("iteration")
        plot2.set_ylabel("hit %")
        plot2.set_title("Learnig curve - genetic - batch size 5")
        # linia trendu \/
        z = numpy.polyfit(domain, accuracy_genetic, 1)
        p = numpy.poly1d(z)
        plot2.plot(domain, p(domain), "r--")
        plt.show()

    if option == "risin_time_comparison":
        # a potem wielu sieci naraz
        # uczymy obie sieci od nowa 100 razy
        time_genetic = []
        time_gradient = []
        domain = range(3)
        for j in domain:
            print(j)
            random.shuffle(net_data)
            # resetowanie sieci
            net = geneticNetwork([7, 4, 4, 2], 50, 5)
            net_normal = network([7, 4, 4, 2], net_gradient, 10)

            start = time.time()
            net.start_learning()
            for i in range(ROWS):
                net.process(net_data[i][0])
                net.correct(net_data[i][1])
            net.stop_learning()

            end = time.time()
            time_genetic.append(end-start)

            start = time.time()
            net_normal.start_learning()
            for i in range(ROWS):
                net_normal.process(net_data[i][0])
                net_normal.correct(net_data[i][1])
            net_normal.stop_learning()

            end = time.time()
            time_gradient.append(end - start)

        plot1 = plt.subplot2grid((2, 1), (0, 0))
        plot1.plot(domain, time_gradient)
        plot1.set_xlabel("attempt")
        plot1.set_ylabel("time needed to learn[s]")
        plot1.set_title("Time spent on learning- gradient")

        plot2 = plt.subplot2grid((2, 1), (1, 0))
        plot2.plot(domain, time_genetic)
        plot2.set_xlabel("attempt")
        plot2.set_ylabel("time needed to learn[s]")
        plot2.set_title("Time spent on learning- genetic")

        plt.show()

    if option == "test_gradient":
        LOOPS = 10
        net_normal = network([7, 4, 3, 2], net_gradient, 1, learnBase=8, fractionLearnRate=3 / 4, learnSuppresion=200)
        error = []
        # tutaj operujemy na input i expected
        net_normal.start_learning()
        for j in range(LOOPS):
            for i in range(ROWS):
                net_normal.process(net_data[i][0])
                net_normal.correct(net_data[i][1])
                error.append(net_normal.total_error(input, expected))
                print(j, i)
        net_normal.stop_learning()

        domain = range(1, ROWS * LOOPS + 1)

        # plt.plot(range(150), error_normal, label="oś x")
        plt.plot((1, 1), (1, 0))
        plt.plot(domain, error, label="price")
        plt.xlabel("iteration")
        plt.ylabel("total error")
        plt.title("Learnig curve - genetic - batch size 5")
        # linia trendu \/
        z = numpy.polyfit(domain, error, 1)
        p = numpy.poly1d(z)
        plt.plot(domain, p(domain), "r--")
        plt.show()

        for i in range(ROWS):
            print(net_normal.process(net_data[i][0]), net_data[i][1])


    if option == "test_genetic":
        LOOPS=3
        net = geneticNetwork([7, 6, 5, 2],net_gradient, 20, 100)
        domain = range(1, ROWS * LOOPS + 1, 10)
        error_genetic = []
        # tutaj operujemy na input i expected
        net.start_learning()
        for j in range(LOOPS):
            for i in range(ROWS):
                net.process(net_data[i][0])
                net.correct(net_data[i][1])
                if(j*LOOPS + i) in domain:
                    error_genetic.append(net.total_error(input, expected))
                print(j, i)
        net.stop_learning()



        # plt.plot(range(150), error_normal, label="oś x")
        #plt.plot((1, 1), (1, 0))
        plt.plot(domain, error_genetic, label="price")
        plt.xlabel("iteration")
        plt.ylabel("total error")
        plt.title("Learnig curve - genetic - batch size 5")
        # linia trendu \/
        z = numpy.polyfit(domain, error_genetic, 1)
        p = numpy.poly1d(z)
        plt.plot(domain, p(domain), "r--")
        plt.show()

        for i in range(ROWS):
            print(net.process(net_data[i][0]), net_data[i][1])

    if option == "test_mix":
        LOOPS = 10
        mix_net = mixNetwork([7, 6, 5, 2], net_gradient, 25, 30, 2, input, expected, 50, learnBase=4,
                             fractionLearnRate=3 / 4, learnSuppresion=400)
        error_genetic = []
        accuracy_mix = []
        domain = range(1, ROWS * LOOPS + 1, 10)
        # tutaj operujemy na input i expected
        mix_net.start_learning()
        for j in range(LOOPS):
            for i in range(ROWS):
                mix_net.process(net_data[i][0])
                mix_net.correct(net_data[i][1])
                if (j * ROWS + i) in domain:
                    error_genetic.append(mix_net.total_error(input, expected))
                    gradient_hits = 0
                    for k in range(ROWS):
                        otp = ParseOutpus(mix_net.process(net_data[k][0]))
                        if Compare_lists(otp, net_data[k][1]):
                            gradient_hits += 1
                    accuracy_mix.append(gradient_hits)
                print(j, i)
        mix_net.stop_learning()

        # plt.plot(range(150), error_normal, label="oś x")
        plt.plot((1, 1), (1, 0))
        plt.plot(domain, error_genetic, label="price")
        plt.plot(domain, accuracy_mix, label="hits")
        plt.xlabel("iteration")
        plt.ylabel("total error")
        plt.title("Learnig curve - mix - batch size 5")
        # linia trendu \/
        z = numpy.polyfit(domain, error_genetic, 1)
        p = numpy.poly1d(z)
        plt.plot(domain, p(domain), "r--")
        plt.show()

        for i in range(ROWS):
            print(mix_net.process(net_data[i][0]), net_data[i][1])
def PlotMusk(option):
    ROWS = 476
    def name_to_tuple_Musk(name):
        if name[0] == 'M':
            return [1]
        else:
            return [0]

    #stworzenie bazy danych
    file = open("Musk/clean1.data")
    csvreader = csv.reader(file)

    table = []
    for row in csvreader:
        table.append(row)

    random.shuffle(table)
    net_data = [None] * ROWS
    input = [None] * ROWS
    expected = [None] * ROWS
    for i in range(ROWS):
        row = table[i]
        name = row[0]
        row = row[2:168]
        for j in range(166):
            row[j] = float(row[j])  # konwersja

        '''# poprawianie do przedziału (0,1)
        row[0] = row[0] / 235047
        row[1] = row[1] / 1000
        row[2] = row[2] / 493
        row[4] = row[4] / 278217
        row[6] = row[6] / 2700
        '''

        input[i] = row
        expected[i] = name_to_tuple_Musk(name)
        net_data[i] = (row, expected[i])
        # print (row,expected)

    if option == "musk_single_try":
        LOOPS=6
        net = geneticNetwork([166, 40, 20, 1],net_gradient, 10, 5)
        net_normal = network([166, 40, 20, 1], net_gradient,10)
        error_genetic = []
        error_normal = []
        #tutaj operujemy na input i expected
        net.start_learning()
        net_normal.start_learning()
        for j in range(LOOPS):
            for i in range(ROWS):
                net.process(net_data[i][0])
                net.correct(net_data[i][1])
                net_normal.process(net_data[i][0])
                net_normal.correct(net_data[i][1])
                error_normal.append(net_normal.total_error(input,expected))
                error_genetic.append(net.total_error(input, expected))
                print(j,i)
        net_normal.stop_learning()
        net.stop_learning()

        domain=range(1,ROWS*LOOPS+1)

        plot1 = plt.subplot2grid((2,1), (0, 0))
        plot1.plot(domain, error_normal, label="price")
        plot1.set_xlabel("iteration")
        plot1.set_ylabel("total error")
        plot1.set_title("Learnig curve - gradient")
        #linia trendu \/
        z = numpy.polyfit(domain, error_normal, 1)
        p = numpy.poly1d(z)
        plot1.plot(domain, p(domain), "r--")

        #plt.plot(range(150), error_normal, label="oś x")
        plot2 = plt.subplot2grid((2, 1), (1, 0))
        plot2.plot(domain, error_genetic, label="price")
        plot2.set_xlabel("iteration")
        plot2.set_ylabel("total error")
        plot2.set_title("Learnig curve - genetic - batch size 5")
        # linia trendu \/
        z = numpy.polyfit(domain, error_genetic, 1)
        p = numpy.poly1d(z)
        plot2.plot(domain, p(domain), "r--")
        plt.show()

        for i in range(ROWS):
            print(net_normal.process(net_data[i][0]),net_data[i][1])
            print(net.process(net_data[i][0]), net_data[i][1])

    if option == "test_generalism":
        def Compare_lists(a,b):
            if len(a)==len(b):
                for i in range(len(a)):
                    if a[i] != b[i]:
                        return False
                return True
            else:
                return False

        def ParseOutpus(list):
            for i in range(len(list)):
                list[i]=round(list[i])
            return list

        LOOPS = 3
        LEARNING_LOOPS = 6
        TEST_PROBES = 50
        # tutaj operujemy na input i expected
        accuracy_gradient = []
        accuracy_genetic = []

        for j in range(LOOPS):
            random.shuffle(net_data)
            net = geneticNetwork([166, 80,40, 20, 1],net_gradient, 10, 10)
            #net_normal = network([166, 80,40, 20, 1], net_gradient, 10, learnBase = 1 ,fractionLearnRate=1, learnSuppresion = 1)
            net.start_learning()
            #net_normal.start_learning()
            for k in range(LEARNING_LOOPS):
                for i in range(ROWS - TEST_PROBES):
                    print(j,k,i)
                    net.process(net_data[i][0])
                    net.correct(net_data[i][1])
                    #net_normal.process(net_data[i][0])
                    #net_normal.correct(net_data[i][1])
            #net_normal.stop_learning()
            net.stop_learning()

            genetic_hits = 0
            gradient_hits = 0
            for i in range(ROWS - TEST_PROBES, ROWS):
                #otp = ParseOutpus(net_normal.process(net_data[i][0]))
                #if Compare_lists(otp, net_data[i][1]):
                #    gradient_hits += 1
                otp = ParseOutpus(net.process(net_data[i][0]))
                if Compare_lists(otp, net_data[i][1]):
                    genetic_hits += 1

            #accuracy_gradient.append(gradient_hits / TEST_PROBES)
            accuracy_genetic.append(genetic_hits / TEST_PROBES)

        domain = range(1, LOOPS + 1)

        plot1 = plt.subplot2grid((2,1), (0, 0))
        #plot1.plot(domain, accuracy_gradient, label="price")
        #plot1.set_xlabel("iteration")
        #plot1.set_ylabel("hit %")
        #plot1.set_title("Learnig curve - gradient")
        #linia trendu \/
        #z = numpy.polyfit(domain, accuracy_gradient, 1)
        #p = numpy.poly1d(z)
        #plot1.plot(domain, p(domain), "r--")

        plot2 = plt.subplot2grid((2, 1), (1, 0))
        plot2.plot(domain, accuracy_genetic, label="price")
        plot2.set_xlabel("iteration")
        plot2.set_ylabel("hit %")
        plot2.set_title("Learnig curve - genetic - batch size 5")
        # linia trendu \/
        z = numpy.polyfit(domain, accuracy_genetic, 1)
        p = numpy.poly1d(z)
        plot2.plot(domain, p(domain), "r--")
        plt.show()

    if option == "musk_time_comparison":
        # a potem wielu sieci naraz
        # uczymy obie sieci od nowa 100 razy
        time_genetic = []
        time_gradient = []
        domain = range(3)
        for j in domain:
            print(j)
            random.shuffle(net_data)
            # resetowanie sieci
            net = geneticNetwork([7, 4, 4, 2], 50, 5)
            net_normal = network([7, 4, 4, 2], net_gradient, 10)

            start = time.time()
            net.start_learning()
            for i in range(ROWS):
                net.process(net_data[i][0])
                net.correct(net_data[i][1])
            net.stop_learning()

            end = time.time()
            time_genetic.append(end-start)

            start = time.time()
            net_normal.start_learning()
            for i in range(ROWS):
                net_normal.process(net_data[i][0])
                net_normal.correct(net_data[i][1])
            net_normal.stop_learning()

            end = time.time()
            time_gradient.append(end - start)

        plot1 = plt.subplot2grid((2, 1), (0, 0))
        plot1.plot(domain, time_gradient)
        plot1.set_xlabel("attempt")
        plot1.set_ylabel("time needed to learn[s]")
        plot1.set_title("Time spent on learning- gradient")

        plot2 = plt.subplot2grid((2, 1), (1, 0))
        plot2.plot(domain, time_genetic)
        plot2.set_xlabel("attempt")
        plot2.set_ylabel("time needed to learn[s]")
        plot2.set_title("Time spent on learning- genetic")

        plt.show()

    if option == "test_gradient":
        LOOPS = 10
        net_normal = network([7, 4, 3, 2], net_gradient, 1, learnBase=8, fractionLearnRate=3 / 4, learnSuppresion=200)
        error = []
        # tutaj operujemy na input i expected
        net_normal.start_learning()
        for j in range(LOOPS):
            for i in range(ROWS):
                net_normal.process(net_data[i][0])
                net_normal.correct(net_data[i][1])
                error.append(net_normal.total_error(input, expected))
                print(j, i)
        net_normal.stop_learning()

        domain = range(1, ROWS * LOOPS + 1)

        # plt.plot(range(150), error_normal, label="oś x")
        plt.plot((1, 1), (1, 0))
        plt.plot(domain, error, label="price")
        plt.xlabel("iteration")
        plt.ylabel("total error")
        plt.title("Learnig curve - genetic - batch size 5")
        # linia trendu \/
        z = numpy.polyfit(domain, error, 1)
        p = numpy.poly1d(z)
        plt.plot(domain, p(domain), "r--")
        plt.show()

        for i in range(ROWS):
            print(net_normal.process(net_data[i][0]), net_data[i][1])


    if option == "test_genetic":
        LOOPS=3
        net = geneticNetwork([166, 80,40, 20, 1],net_gradient, 10, 10)
        domain = range(1, ROWS * LOOPS + 1, 10)
        error_genetic = []
        # tutaj operujemy na input i expected
        net.start_learning()
        for j in range(LOOPS):
            for i in range(ROWS):
                net.process(net_data[i][0])
                net.correct(net_data[i][1])
                if(j*LOOPS + i) in domain:
                    error_genetic.append(net.total_error(input, expected))
                print(j, i)
        net.stop_learning()



        # plt.plot(range(150), error_normal, label="oś x")
        #plt.plot((1, 1), (1, 0))
        plt.plot(domain, error_genetic, label="price")
        plt.xlabel("iteration")
        plt.ylabel("total error")
        plt.title("Learnig curve - genetic - batch size 5")
        # linia trendu \/
        z = numpy.polyfit(domain, error_genetic, 1)
        p = numpy.poly1d(z)
        plt.plot(domain, p(domain), "r--")
        plt.show()

        for i in range(ROWS):
            print(net.process(net_data[i][0]), net_data[i][1])
