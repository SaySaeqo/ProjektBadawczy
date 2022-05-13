import numpy as np
from constants import *
from utils import sigmoid, sigmoid_der


def average(networks_array):
    """
    Computates average of the Networks' array

    :param networks_array: list of Network objects
    :return: Network which store average by weights and biases from Networks' array
    """
    my_sum = Network.create_empty(networks_array[0].model_shape)
    for network in networks_array:
        my_sum += network
    return my_sum / len(networks_array)

class Network:
    """
    ===========
        Attributes
    ===========
    layers : list of layers in network
        Each layer contains matrices for layer's weights, biases, values and zs in exactly that order

    ===========
        Methods
    ===========
    process(args)
        compute values from input
    correct(inputs, expected_outputs)
        use gradient algorithm to train Network
    Network.create_empty()
        create Network with all matrices of layers with values 0

    operators (affects each neuron)
        * += Network
        * -= Network
        * / number
    """

    def __init__(self, model_shape, correct_func=None, empty=False):
        """
        :param model_shape: array of layers' sizes, where size of array is number of layers,
            like [3, 5, 3] that means 3 layers, 1st has 3 neurons, 2nd - 5 neurons and 3rd - 3 neurons
        """
        self.correct_func = correct_func
        self.model_shape = model_shape
        model_shape_shifted = ([0] + model_shape)[:-1]
        self.nb_layers = len(model_shape)

        if empty:
            weights = [np.zeros((nb_neur_prev, nb_neur_cur))
                       for nb_neur_prev, nb_neur_cur in zip(model_shape_shifted, model_shape)]
            biases = [np.zeros((nb_neur, 1)) for nb_neur in model_shape]
        else:
            weights = [np.random.rand(nb_neur_prev, nb_neur_cur)
                       for nb_neur_prev, nb_neur_cur in zip(model_shape_shifted, model_shape)]
            biases = [np.random.rand(nb_neur, 1) for nb_neur in model_shape]

        values = [np.zeros((1, nb_neur)) for nb_neur in model_shape]
        zs = [np.zeros((nb_neur, 1)) for nb_neur in model_shape]
        self.layers = zip(weights, biases, values, zs)
        self.layers = [list(layer) for layer in self.layers]

    def __call__(self, *args, **kwargs):
        return self.process(*args)

    def process(self, args):
        """
        Activate network to compute results from args.
        All activation values from all neurons all stored in values parameter.

        :param args: collection of arguments in number of 1st layer size
        :return: collection of results in number of last layer size
        """

        # turn on the net
        first = True
        v_prev = None
        for layer in self.layers:
            if first:
                v_prev = layer[2] = np.matrix(args)
                first = False
                continue

            w, b, v, z = layer
            layer[3] = np.transpose(v_prev @ w) + b
            v_prev = layer[2] = np.transpose(sigmoid(layer[3]))

        return v_prev

    def correct(self, inputs, expected_outputs):
        return self.correct_func(self, inputs, expected_outputs)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        """
        -------LAYER 1-------
        | N1 | N2 | N3 |...       Layer1
        -------LAYER 2-------
        ...                       Layer2
        -------LAYER ...-------
        ...                        ...
        ---------END---------

        :return: string like above
        """
        return f"{self.layers}"

    @classmethod
    def create_empty(cls, model_shape):
        return cls(model_shape, empty=True)

    def __iadd__(self, other):
        for layer1, layer2 in zip(self.layers, other.layers):
            w1, b1, v1, z1 = layer1
            w2, b2, v2, z2 = layer2
            w1 += w2
            b1 += b1
        return self

    def __isub__(self, other):
        for layer1, layer2 in zip(self.layers, other.layers):
            w1, b1, v1, z1 = layer1
            w2, b2, v2, z2 = layer2
            w1 -= w2
            b1 -= b1
        return self

    def __truediv__(self, other):
        result = Network.create_empty(self.model_shape)
        for result_layer, layer1 in zip(result.layers, self.layers):
            w, b, v, z = layer1
            result_layer[0] = w / other
            result_layer[1] = b / other
        return result
