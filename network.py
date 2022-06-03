import copy
import itertools
import operator
import random

import numpy as np
from constants import *
from utils import sigmoid, sigmoid_der, progress_bar


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
        :param correct_func: function which is used to train Network.
        """
        self.correct_func = correct_func
        self.model_shape = model_shape
        self.nb_layers = len(model_shape) - 1

        if empty:
            weights = [np.zeros((nb_neur_prev, nb_neur_cur))
                       for nb_neur_prev, nb_neur_cur in zip(model_shape, model_shape[1:])]
            biases = [np.zeros((nb_neur, 1)) for nb_neur in model_shape[1:]]
        else:
            weights = [np.random.rand(nb_neur_prev, nb_neur_cur)
                       for nb_neur_prev, nb_neur_cur in zip(model_shape, model_shape[1:])]
            biases = [np.random.rand(nb_neur, 1) for nb_neur in model_shape[1:]]

        values = [np.zeros((1, nb_neur)) for nb_neur in model_shape[1:]]
        zs = [np.zeros((nb_neur, 1)) for nb_neur in model_shape[1:]]
        self.layers = [{"w": w, "b": b, "v": v, "z": z} for w, b, v, z in zip(weights, biases, values, zs)]

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
        v_prev = np.matrix(args)
        for layer in self.layers:
            layer["z"] = np.transpose(v_prev @ layer["w"]) + layer["b"]
            v_prev = layer["v"] = np.transpose(sigmoid(layer["z"]))

        return v_prev.tolist()[0]

    def train(self, inputs, targets, **kwargs):
        if isinstance(self.correct_func, list):
            history = {
                "all_costs": [],
                "av_costs": [],
                "success_rate": [],
                "change_points": []
            }
            idx = 0
            while True:
                h = self.correct_func[idx % len(self.correct_func)](self, inputs, targets, **kwargs)
                history["all_costs"] += h["all_costs"]
                history["av_costs"] += h["av_costs"]
                history["success_rate"] += h["success_rate"]
                if history["success_rate"][-1] > 0.8 or idx >= 10:
                    break
                history["change_points"] += [len(history["av_costs"])]
                idx += 1
            return history

        return self.correct_func(self, inputs, targets, **kwargs)

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
        for i in range(self.nb_layers):
            text += f"-------LAYER {i + 1}-------\n"
            text += f"WEIGHTS:\n{self.layers[i]['w']}\n"
            text += f"BIASES:\n{self.layers[i]['b']}\n"
            text += f"VALUES:\n{self.layers[i]['v']}\n"
        text += "---------END---------"
        return text

    @classmethod
    def create_empty(cls, model_shape):
        """
        Creates Network with zeros instead of random values for biases and weights.
        """
        return cls(model_shape, empty=True)

    def __iadd__(self, other):
        for layer1, layer2 in zip(self.layers, other.layers):
            layer1["w"] += layer2["w"]
            layer1["b"] += layer2["b"]
        return self

    def __isub__(self, other):
        for layer1, layer2 in zip(self.layers, other.layers):
            layer1["w"] -= layer2["w"]
            layer1["b"] -= layer2["b"]
        return self

    def __truediv__(self, other):
        result = Network.create_empty(self.model_shape)
        for result_layer, layer1 in zip(result.layers, self.layers):
            result_layer["w"] = layer1["w"] / other
            result_layer["b"] = layer1["b"] / other
        return result

    def __iter__(self):
        self._iter_layer = 0
        self._iter_neuron = 0
        return self

    def __next__(self):
        if self._iter_layer < self.nb_layers:
            if self._iter_neuron >= self.layers[self._iter_layer]["v"].size:
                self._iter_layer += 1
                self._iter_neuron = 0
                return next(self)
            neuron = {
                "w": self.layers[self._iter_layer]["w"][:, self._iter_neuron],
                "b": self.layers[self._iter_layer]["b"][0, self._iter_neuron],
                "v": self.layers[self._iter_layer]["v"][0, self._iter_neuron],
                "z": self.layers[self._iter_layer]["z"][0, self._iter_neuron]
            }
            return neuron
        else:
            raise StopIteration


class WLTNetwork(Network):
    def __init__(self, model_shape, correct_func=None, empty=False):
        super().__init__(model_shape, correct_func, empty)
        self.activation_genome = [1 for _ in range(sum(model_shape[1:]))]
        self.weights_genome = [np.ones((nb_neur_prev, nb_neur_cur))
                               for nb_neur_prev, nb_neur_cur in zip(model_shape, model_shape[1:])]

    def process(self, args):
        """
        Activate network to compute results from args.
        All activation values from all neurons all stored in values parameter.

        :param args: collection of arguments in number of 1st layer size
        :return: collection of results in number of last layer size
        """

        # activation genome handlers
        curr_gen = 0
        gen_length = 0

        # turn on the net
        v_prev = np.matrix(args)
        for layer, weights_genome_layer in zip(self.layers, self.weights_genome):
            gen_length = layer["w"].size

            w = np.multiply(layer["w"], weights_genome_layer)
            layer["z"] = np.transpose(v_prev @ w) + layer["b"]
            v_prev = layer["v"] = np.transpose(sigmoid(layer["z"]))
            curr_gen += gen_length

        return v_prev.tolist()[0]
