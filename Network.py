import numpy as nmp
from Constants import *

from utils import sigmoid


class Neuron:
    def __init__(self, type, ascendants_number=0):
        self.type = type
        self.value = None
        if type != INPUT:
            self.bias = nmp.random.rand()
            self.ascendants_weights = nmp.random.rand(ascendants_number)
        else:
            self.bias = None
            self.ascendants_weights = None

    def __str__(self):
        return f"Neuron({self.type}, {len(self.ascendants_weights)})"

    def __repr__(self):
        return f"Bias:{self.bias} A_Weights:{self.ascendants_weights}"


class Network:
    """
    matrices-macierze wag. wagi itej warstwy są na itej pozycji (pierwsza element tej tablicy to None, bo zerowa
    warstwa nie ma wag)
    biases = wektory biasów analogicznie jak macierze wag
    arguments
    function
    correct_function
    """

    def __init__(self, nrn_nmb, correct_function):
        """
        :param nrn_nmb: neurons number per layer, czytając od lewej
        :param correct_function: funkcja ucząca/poprawiająca sieć
        """

        self.neurons = [[[Neuron(INPUT)] * nrn_nmb[0]]]
        for i in range(1, len(nrn_nmb)):
            self.neurons += [[Neuron(OUTPUT, nrn_nmb[i - 1])] * nrn_nmb[i]]

        self.correct_function = correct_function

    def process(self, args):
        """
        liczy wartości aktywacji WSZYSTKICH neuronów, zwraca tylko ostatnią warstwę
        ale wszystkie wartości są przechowywane wewnątrz obiektu w polu neuron_values!
        :param args:
        :return:
        """
        for i in range(len(args)):
            self.neurons[0][i].value = args[i]

        for layer_idx in range(1, len(self.neurons)):
            prev_values = [neuron.values for neuron in self.neurons[layer_idx-1]]
            for neuron in self.neurons[layer_idx]:
                neuron.value = nmp.matmul(neuron.ascendants_weights, prev_values)
                neuron.value += neuron.bias

        return [neuron.values for neuron in self.neurons[-1]]

    def correct(self, expected_result):
        """
        Funkcja zmieniająca wartości wag i biasów, w zależności od funkcji correct zdefiniowanej
        w konstruktorze. Funkcja correct, musi być poprzedzona wywołaniem process
        :param expected_result: tablica wartości neuronów ostatniej warstwy
        :return: 2 obiekty: gradient wag, gradient biasów
        """

        matrices_der, biases_der = self.correct_function(self.neuron_values, expected_result, self)
        # ulepsz każdą warstwę
        for i in range(1, self.layers):
            self.weights[i] = nmp.subtract(self.weights[i], matrices_der[i])
            self.biases[i] = nmp.subtract(self.biases[i], biases_der[i])
        return matrices_der, biases_der

    def __str__(self):
        """ahh"""
        text = ""
        # wypisz każdą macierz
        for i in range(self.layers):
            # OZDOBA górna krawedź/
            text += ' '
            for j in range(len(self.weights[i][0])):
                text += ' ----'
            # OZDOBA górna krawedź\

            # każdy wiersz
            text += '\n'
            for j in range(self.nrn_nmb[i]):
                text += '| '
                # wypisz kolumne
                for k in range(len(self.weights[i][j])):
                    if self.weights[i][j][k] != 'x':
                        text += f"{self.weights[i][j][k]:.1f} ".rjust(5)
                    else:
                        text += f"  {self.weights[i][j][k]}  "

                text += '|\n'

            # OZDOBA dolna krawedź/
            text += ' '
            for k in range(len(self.weights[i][j])):
                text += ' ----'
            text += '\n'
            # OZDOBA dolna krawedź\

            text += '\n'
        return text
