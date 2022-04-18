from typing import Union, List, Any
import string
import numpy as nmp

class network:
    '''
    matrices-macierze wag. wagi itej warstwy są na itej pozycji (pierwsza element tej tablicy to None, bo zerowa warstwa nie ma wag)
    biases = wektory biasów analogicznie jak macierze wag
    arguments
    function
    correct_function
    '''

    def __init__(self,layers,nrn_nmb,correct_function):
        '''
        :param layers:
        :param nrn_nmb: neurons number per layer, czytając od lewej
        '''

        self.layers=layers
        self.matrices=[None]*(layers)
        self.biases=[None]*(layers)
        self.nrn_nmb=nrn_nmb
        self.neuron_values=[None]*(layers)
        self.arguments=[]
        self.function = None
        self.correct_function = correct_function
        self.learn_rate=0.1

        #utworzenie tablicy macierzy wag
        matrix=self.create_X_matrix(nrn_nmb[0],1)
        self.matrices[0] = matrix
        for i in reversed(range(1, layers)):
            matrix=self.create_rand_matrix(nrn_nmb[i],nrn_nmb[i-1])
            self.matrices[i]=matrix

        #utworzenie tablicy bias
        for i in reversed(range(1, layers)):
            bias=[0]*(nrn_nmb[i])
            self.biases[i]=bias

        #konwersja macierzy i biasów na tablice argumentów


    def process(self,args):
        '''
        liczy wartości aktywacji WSZYSTKICH neuronów, zwraca tylko ostatnią warstwę
        ale wszystkie wartości są przechowywane wewnątrz obiektu!
        :param args:
        :return:
        '''
        self.neuron_values[0] = args;
        for i in range(1,self.layers):
            self.neuron_values[i] = nmp.matmul(self.matrices[i], self.neuron_values[i-1])
            self.neuron_values[i] = nmp.add(self.neuron_values[i-1], self.biases[i])
        return self.neuron_values[self.layers - 1]


    def correct(self,expected_result):

        matrices_der,biases_der=self.correct_function(self.neuron_values,expected_result,self)
        print(matrices_der)
        #zmien potem oryginały

    def create_rand_matrix(self,x,y):
        ''' tworzy macierz z losowymi wartościami o wymiarach x wierszy na y kolumn'''

        matrix=[[1]*y]*x # póki co nie losowe XD
        return matrix

    def create_X_matrix(self,x,y):
        ''' tworzy macierz z losowymi wartościami o wymiarach x wierszy na y kolumn'''

        matrix=[['x']*y]*x # póki co nie losowe XD
        return matrix

    def __str__(self):
        text=""
        #wypisz każdą macierz
        for i in range(self.layers):
            # OZDOBA górna krawedź/
            text += ' '
            for j in range(len(self.matrices[i][0])):
                text += ' -'
            # OZDOBA górna krawedź\

            #każdy wiersz
            text += '\n'
            for j in range(self.nrn_nmb[i]):
                text += '| '
                #wypisz kolumne
                for k in range(len(self.matrices[i][j])):
                    text+=  str(self.matrices[i][j][k]) + ' '

                text += '|\n'

            # OZDOBA dolna krawedź/
            text += ' '
            for k in range(len(self.matrices[i][j])):
                text+= ' -'
            text += '\n'
            # OZDOBA dolna krawedź\

            text+='\n'
        return text
