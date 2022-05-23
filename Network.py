import copy
import math

from Functions import sigmoid
import numpy as nmp
import random
from Functions import sigmoid


def cost(result, expected):
    sum = 0
    for i in range(len(result)):
        sum += (result[i] - expected[i]) ** 2
    return sum


class network:
    '''
    matrices-macierze wag. wagi itej warstwy są na itej pozycji (pierwsza element tej tablicy to None, bo zerowa
    warstwa nie ma wag)
    biases = wektory biasów analogicznie jak macierze wag
    arguments
    function
    correct_function
    '''

    def __init__(self, nrn_nmb, correct_function, batch_size, learnBase = 10 , fractionLearnRate= 3/4, learnSuppresion = 1):
        """
        :param nrn_nmb: neurons number per layer, czytając od lewej
        :param correct_function: funkcja ucząca/poprawiająca sieć
        """

        self.layers = len(nrn_nmb)
        self.matrices = [None] * (self.layers)
        self.biases = [None] * (self.layers)
        self.nrn_nmb = nrn_nmb
        self.neuron_values = [None] * (self.layers)
        self.neuron_gradients = [[None]* neurons for neurons in nrn_nmb]
        self.correct_function = correct_function
        self.batch_size = batch_size
        self.batch_iteration = 0
        self.batch_matrix_grad = self.create_zero_matrices_batch()
        self.batch_bias_grad = self.create_zero_biases_batch()
        self.learnBase = learnBase
        self.fractionLearnRate = fractionLearnRate
        self.learnCounter = 1
        self.learnSuppresion = learnSuppresion

        # utworzenie tablicy macierzy wag
        matrix = self.create_X_matrix(nrn_nmb[0], 1)
        self.matrices[0] = matrix
        for i in reversed(range(1, self.layers)):
            matrix = self.create_rand_matrix(nrn_nmb[i], nrn_nmb[i - 1])
            self.matrices[i] = matrix

        # utworzenie tablicy bias
        for i in reversed(range(1, self.layers)):
            # bias=[0]*(nrn_nmb[i])
            bias = nmp.random.rand(nrn_nmb[i])
            self.biases[i] = bias

        # konwersja macierzy i biasów na tablice argumentów

    def process(self, args):
        """
        liczy wartości aktywacji WSZYSTKICH neuronów, zwraca tylko ostatnią warstwę
        ale wszystkie wartości są przechowywane wewnątrz obiektu w polu neuron_values!
        :param args:
        :return:
        """
        self.neuron_values[0] = args
        for i in range(1, self.layers):
            self.neuron_values[i] = nmp.matmul(self.matrices[i], self.neuron_values[i - 1])
            self.neuron_values[i] = nmp.add(self.neuron_values[i], self.biases[i])
            for j in range(self.nrn_nmb[i]):
                self.neuron_values[i][j] = sigmoid(self.neuron_values[i][j])
        return self.neuron_values[self.layers - 1]

    def correct(self, expected_result):
        """
        Funkcja zmieniająca wartości wag i biasów, w zależności od funkcji correct zdefiniowanej
        w konstruktorze. Funkcja correct, musi być poprzedzona wywołaniem process
        :param expected_result: tablica wartości neuronów ostatniej warstwy
        :return: 2 obiekty: gradient wag, gradient biasów
        """

        self.batch_iteration += 1

        matrices_der, biases_der = self.correct_function(self.neuron_values, expected_result, self)
        # ulepsz każdą warstwę

        #batchowanie
        for i in range(1, self.layers):
            #matrices_der[i] = nmp.multiply(matrices_der[i], self.learnBase * ( 1 / self.learnCounter) ** self.learnSuppresion)  # skalowanie z learnratem
            matrices_der[i] = nmp.multiply(matrices_der[i], self.learnBase * (self.fractionLearnRate) ** (self.learnCounter/self.learnSuppresion) ) # skalowanie z learnratem
            #biases_der[i] = nmp.multiply(biases_der[i], self.learnBase * (1 / self.learnCounter) ** self.learnSuppresion)  # skalowanie z learnratem
            biases_der[i] = nmp.multiply(biases_der[i],  self.learnBase * (self.fractionLearnRate) ** (self.learnCounter/self.learnSuppresion))  # skalowanie z learnratem


            self.batch_matrix_grad[i] = nmp.add(self.batch_matrix_grad[i], matrices_der[i])

            self.batch_bias_grad[i] = nmp.add(self.batch_bias_grad[i], biases_der[i])



        #update wag po zapełnieniu batchu
        if self.batch_iteration == self.batch_size:
            # matrices_der, biases_der = self.correct_function(self.neuron_values, expected_result, self)

            "uśrednianie"
            for matrix in self.batch_matrix_grad[1:]:
                for row in matrix:
                    for val in row:
                        val = val / self.batch_size # uśrednianie

            for layer in self.batch_bias_grad[1:]:
                for val in layer:
                    val = val / self.batch_size # uśrednianie



            for i in range(1, self.layers):
                self.matrices[i] = nmp.subtract(self.matrices[i], self.batch_matrix_grad[i])
                self.biases[i] = nmp.subtract(self.biases[i],  self.batch_bias_grad[i])

            self.batch_matrix_grad = self.create_zero_matrices_batch()
            self.batch_bias_grad = self.create_zero_biases_batch()
            self.batch_iteration = 0

        self.learnCounter += 1
        return matrices_der, biases_der

    def asses(self, expected_result):
        sum = 0
        for i in range(len(self.neuron_values[self.layers - 1])):
            sum += (self.neuron_values[self.layers - 1][i] - expected_result[i]) ** 2
            # sum+=abs(self.neuron_values[self.layers-1][i]-expected_result[i])
        return sum

    def create_rand_matrix(self, x, y):
        """ tworzy macierz z losowymi wartościami o wymiarach x wierszy na y kolumn"""

        matrix = nmp.random.rand(x, y)
        return matrix

    def create_X_matrix(self, x, y):
        """ tworzy macierz X'ów o wymiarach x wierszy na y kolumn
            głównie dla warstwy wejściowej, dla której nie ma wag """

        matrix = [['x'] * y] * x  # póki co nie losowe XD
        return matrix

    def create_zero_matrices_batch(self):
        matrices=[None]*self.layers
        for i in reversed(range(1, self.layers)):
            matrices[i] = nmp.zeros((self.nrn_nmb[i], self.nrn_nmb[i - 1]))
        return matrices

    def create_zero_biases_batch(self):
        biases=[None]*self.layers
        for i in reversed(range(1, self.layers)):
            biases[i] = nmp.zeros(self.nrn_nmb[i])
        return biases

    def start_learning(self):
        '''bedzie to mówiło, że podczas process, '''
        self.learning = True

    def stop_learning(self):
        self.learning = False

    def total_error(self, input, expected_result):
        error = 0
        was_learning = False
        if (self.learning == True):
            was_learning = True
            self.learning = False
        for i in range(len(input)):
            result = self.process(input[i])
            error += cost(result, expected_result[i])
        if (was_learning == True):
            self.learning = True
        return error

    def __str__(self):
        """ahh"""
        text = ""
        # wypisz każdą macierz
        for i in range(self.layers):
            # OZDOBA górna krawedź/
            text += ' '
            for j in range(len(self.matrices[i][0])):
                text += ' ----'
            # OZDOBA górna krawedź\

            # każdy wiersz
            text += '\n'
            for j in range(self.nrn_nmb[i]):
                text += '| '
                # wypisz kolumne
                for k in range(len(self.matrices[i][j])):
                    if self.matrices[i][j][k] != 'x':
                        text += f"{self.matrices[i][j][k]:.1f} ".rjust(5)
                    else:
                        text += f"  {self.matrices[i][j][k]}  "

                text += '|\n'

            # OZDOBA dolna krawedź/
            text += ' '
            for k in range(len(self.matrices[i][j])):
                text += ' ----'
            text += '\n'
            # OZDOBA dolna krawedź\

            text += '\n'
        return text


class geneticNetwork(network):
    '''
    pole nets zawiera wszystkie sieci
    net_numb ich liczbe
    '''

    def __init__(self, nrn_nmb,correct_function, net_numb, batch_size, learnBase = 10 , fractionLearnRate= 3/4, learnSuppresion = 1):
        self.learning = False
        self.net_numb = net_numb
        self.correct_function = correct_function
        self.nets = [None] * net_numb
        self.args = []
        self.nrn_nmb = nrn_nmb
        self.expected_result = []
        self.last_args = []
        self.last_expected = []
        self.batch_size=batch_size
        self.learnBase = learnBase
        self.fractionLearnRate = fractionLearnRate
        self.learnCounter = 1
        self.learnSuppresion = learnSuppresion

        for i in range(net_numb):
            self.nets[i] = network(nrn_nmb, None, batch_size)

    def process(self, args):
        self.args = args
        """
        wywołuje tę samą metodę co siec, lecz dla tej o indeksie zero.
        tam powinna być najskuteczniejsza
        """
        if self.learning:
            for net in self.nets:
                net.process(args)
            return self.nets[0].process(args)
        else:
            return self.nets[0].process(args)

    def average_net_weight(self, net1, net2):
        '''pierwsza próba krzyżowania'''
        net = network(self.nrn_nmb, None)
        for layer in range(1, net1.layers):
            for row in range(len(net1.matrices[layer])):
                for col in range(len(net1.matrices[layer][row])):
                    net.matrices[layer][row][col] = (net1.matrices[layer][row][col] + net2.matrices[layer][row][
                        col]) / 2
        return net

    def radius_cross(self, net1, net2):
        net = copy.deepcopy(net1)
        #matrices_der, biases_der = self.correct_function(net.neuron_values, self.expected_result, net1)

        for layer in range(1, net1.layers):
            for row in range(len(net1.matrices[layer])):
                for col in range(len(net1.matrices[layer][row])):
                    distance = abs(net1.matrices[layer][row][col] - net2.matrices[layer][row][col])
                    net.matrices[layer][row][col] += random.uniform(-1, 1) * distance
                    #net.matrices[layer][row][col] += random.uniform(-1, 1) * distance * matrices_der[layer][row][col]
                    #net.matrices[layer][row][col] += distance * random.uniform(-1, 1)  * self.learnBase * (self.fractionLearnRate) ** (self.learnCounter/self.learnSuppresion)
                distance = abs(net1.biases[layer][row] - net2.biases[layer][row])
                net.biases[layer][row] += random.uniform(-1, 1) * distance
                #net.biases[layer][row] += random.uniform(-1, 1) * distance * biases_der[layer][row]
                #net.biases[layer][row] += distance * random.uniform(-1, 1) * self.learnBase * (self.fractionLearnRate) ** (self.learnCounter/self.learnSuppresion)

        return net

    def cross_nets(self):
        nets = copy.deepcopy(self.nets)
        children = []
        for i in range(math.floor(self.net_numb / 2)):
            parents = random.sample(nets, 2)


            nets.remove(parents[0])
            nets.remove(parents[1])

            # usrednianie argumentow + tworzenie nowej sieci potomnej
            # avg_net = self.average_net_weight(parents[0], parents[1])
            avg_net = self.radius_cross(parents[0], parents[1])

            children.append(avg_net)
        # zwraca krzyżówki
        return children

    def mutate_nets(self, crossed):
        '''mutuje potencjalnie wszystkie sieci'''
        for net in crossed:
            for layer in range(1, net.layers):
                for row in range(len(net.matrices[layer])):
                    # modyfikacja wag
                    for col in range(len(net.matrices[layer][row])):
                        if random.randint(0, 100) > 85:
                            delta = 0.1
                            if random.randint(0, 1) == 1:
                                net.matrices[layer][row][col] += delta
                            else:
                                net.matrices[layer][row][col] -= delta
                    # mutacja biasów
                    if random.randint(0, 100) > 85:
                        delta = 0.1
                        if random.randint(0, 1) == 1:
                            net.biases[layer][row] += delta
                        else:
                            net.biases[layer][row] -= delta

    def asses_nets(self):
        assesment = [0] * self.net_numb
        for i in self.net_numb:
            assesment[i] = self.nets[i].asses(self.expected_result)

    def sort_nets(self, nets):
        # assesments=self.asses_nets()

        #wpierw ocena nowego wejścia
        #nets_assesments = [(net, net.asses(self.expected_result)) for net in nets]
        nets_assesments = [None]*len(nets)
        #potem dodajemy stare wejscie
        for j in range(len(nets)):
            sum=0
            for i in range(min(len(self.last_args),self.batch_size)):
                nets[j].process(self.last_args[i])
                sum+=nets[j].asses(self.last_expected[i])
            nets_assesments[j] = (nets[j],sum)
        # sorted_nets = sorted(nets, key=lambda x: x.asses(self.expected_result), reverse=False)
        sorted_nets = sorted(nets_assesments, key=lambda x: x[1], reverse=False)
        sorted_nets = [net[0] for net in sorted_nets]
        return sorted_nets

    def select_best(self, crossed):
        nets = self.nets.copy() + crossed.copy()
        sorted = self.sort_nets(nets)
        best = sorted[:len(nets) - len(crossed)]
        return best

    def correct(self, expected_result):
        """
        correct genetyczny
        """

        self.expected_result = expected_result

        prev_best = copy.deepcopy(self.nets[:2]) #tutaj był najpewniej błąd referencyjny

        # wpierw symulowanie każdej sieci
        #for net in self.nets:
        #    net.process(self.args)

        # i teraz genetyka
        crossed = self.cross_nets()
        #for net in crossed:
        #    net.process(self.args)

        self.mutate_nets(crossed)
        all = crossed + prev_best
        self.nets = self.select_best(all)

        self.last_expected.append(expected_result)
        self.last_args.append(self.args)
        if(len(self.last_expected) == self.batch_size+1):
            self.last_expected=self.last_expected[1:self.batch_size]
            self.last_args = self.last_args[1:self.batch_size]


    def total_error(self, input, expected_result):
        error = 0
        was_learning = False
        if (self.learning == True):
            was_learning = True
            self.nets[0].stop_learning()
        for i in range(len(input)):
            result = self.nets[0].process(input[i])
            error += cost(result, expected_result[i])

        if (was_learning == True):
            self.nets[0].start_learning()

        return error

    def print_assesments(self, expected=[]):
        if expected == []:
            for net in self.nets:
                print(net.asses(self.expected_result))
        else:
            for net in self.nets:
                print(net.asses(expected))


    def __str__(self):

        return self.nets[0].__str__()