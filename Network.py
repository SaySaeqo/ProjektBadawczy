from utils import sigmoid
import numpy as nmp


class network:
    '''
    matrices-macierze wag. wagi itej warstwy są na itej pozycji (pierwsza element tej tablicy to None, bo zerowa
    warstwa nie ma wag)
    biases = wektory biasów analogicznie jak macierze wag
    arguments
    function
    correct_function
    '''

    def __init__(self, nrn_nmb, correct_function):
        """
        :param nrn_nmb: neurons number per layer, czytając od lewej
        :param correct_function: funkcja ucząca/poprawiająca sieć
        """

        self.layers = len(nrn_nmb)
        self.matrices = [None] * (self.layers)
        self.biases = [None] * (self.layers)
        self.nrn_nmb = nrn_nmb
        self.neuron_values = [None] * (self.layers)
        self.correct_function = correct_function

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

        matrices_der, biases_der = self.correct_function(self.neuron_values, expected_result, self)
        # ulepsz każdą warstwę
        for i in range(1, self.layers):
            self.matrices[i] = nmp.subtract(self.matrices[i], matrices_der[i])
            self.biases[i] = nmp.subtract(self.biases[i], biases_der[i])
        return matrices_der, biases_der

    def create_rand_matrix(self, x, y):
        """ tworzy macierz z losowymi wartościami o wymiarach x wierszy na y kolumn"""

        matrix = nmp.random.rand(x, y)
        return matrix

    def create_X_matrix(self, x, y):
        """ tworzy macierz X'ów o wymiarach x wierszy na y kolumn
            głównie dla warstwy wejściowej, dla której nie ma wag """

        matrix = [['x'] * y] * x  # póki co nie losowe XD
        return matrix

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
