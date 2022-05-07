import numpy as np
from Constants import *
from utils import sigmoid


class Neuron:
    """
    ===========
        Attributes
    ===========
    bias : float

    weights : list
        weights representing edges from predecessors in net
    value : float
        computed when net process inputs, equals sigmoid(z)
    z : float
        helper for correct function, equals weights*predecessors_values + bias

    ==========
        Methods
    ==========
    operators (affects only weights and bias)
        *   \+ Neuron
        *   += Neuron
        *   \- Neuron
        *   -= Neuorn
        *   / number
        *   /= number
    """

    def __init__(self, amount_of_predecessors=0):
        self.weights = np.random.rand(amount_of_predecessors)  # weights from predecessors
        self.bias = np.random.rand()
        self.value = 0  # computed on network process
        self.z = 0  # value before calling sigmoid on calcs with weights and bias

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        """
        Bias:Weights

        :return: string like above
        """
        return f"{self.bias}:{self.weights}"

    def __add__(self, other):
        result = Neuron()
        result.weights = np.add(self.weights, other.weights)
        result.bias = self.bias + other.bias
        return result

    def __iadd__(self, other):
        self.weights = np.add(self.weights, other.weights)
        self.bias += other.bias
        return self

    def __sub__(self, other):
        result = Neuron()
        result.weights = np.subtract(self.weights, other.weights)
        result.bias = self.bias - other.bias
        return result

    def __isub__(self, other):
        self.weights = np.subtract(self.weights, other.weights)
        self.bias -= other.bias
        return self

    def __truediv__(self, other):
        result = Neuron()
        result.weights = np.divide(self.weights, other)
        result.bias = self.bias / other
        return result

    def __itruediv__(self, other):
        self.weights = np.divide(self.weights, other)
        self.bias /= other
        return self


class Network:
    """
    ===========
        Attributes
    ===========
    layers : list of layers in network
        Each layer contains list of neurons in number described in model_shape.

    ===========
        Methods
    ===========
    process(args)
        compute values from input
    Network.create_empty()
        create Network with empty layers (without neurons, layers with len of 0)

    operators (affects each neuron)
        * \+ Network
        * += Network
        * / number
    """

    def __init__(self, model_shape):
        """
        :param model_shape: array of layers' sizes, where size of array is number of layers,
            like [3, 5, 3] that means 3 layers, 1st has 3 neurons, 2nd - 5 neurons and 3rd - 3 neurons
        :param correct_function: function used to improve network
        """

        self.model_shape = model_shape
        model_shape_shifted = ([0] + model_shape)[:-1]
        self.layers = [[Neuron(bef) for _ in range(cur)]
                       for bef, cur in zip(model_shape_shifted, model_shape)]

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
        values = args
        for layer in self.layers:
            if first:

                # load first layer
                for neuron, arg in zip(layer, args):
                    neuron.value = arg

                first = False
                continue

            # activate neurons
            for neuron in layer:
                neuron.z = sum(w * v for w, v in zip(neuron.weights, values))
                neuron.z += neuron.bias
                neuron.value = sigmoid(neuron.z)

            values = (neuron.value for neuron in layer)

        return list(values)

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
        text = ""
        for i in range(len(self.layers)):
            text += f"-------LAYER {i}-------\n"
            text += "| "
            for neuron in self.layers[i]:
                text += f"{neuron} | "
            text += "\n"
        text += "---------END---------"
        return text

    @classmethod
    def create_empty(cls, model_shape):
        return cls([0 for layer in model_shape])

    def __add__(self, other):
        result = Network.create_empty(self.model_shape)
        for result_layer, layer1, layer2 in zip(result.layers, self.layers, other.layers):
            result_layer += [neuron1 + neuron2 for neuron1, neuron2 in zip(layer1, layer2)]
        return result

    def __iadd__(self, other):
        for layer1, layer2 in zip(self.layers, other.layers):
            for neuron1, neuron2 in zip(layer1, layer2):
                neuron1 += neuron2
        return self

    def __truediv__(self, other):
        result = Network.create_empty(self.model_shape)
        for result_layer, layer1 in zip(result.layers, self.layers):
            result_layer += [neuron / other for neuron in layer1]
        return result
