import numpy as np
from constants import *
from utils import sigmoid, sigmoid_der


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

    def __init__(self, model_shape, empty = False):
        """
        :param model_shape: array of layers' sizes, where size of array is number of layers,
            like [3, 5, 3] that means 3 layers, 1st has 3 neurons, 2nd - 5 neurons and 3rd - 3 neurons
        """
        self.model_shape = model_shape
        model_shape_shifted = ([0] + model_shape)[:-1]
        self.nb_layers = len(model_shape)

        if empty:
            weights = [np.zeros((nb_neur_prev, nb_neur_cur))
                       for nb_neur_prev, nb_neur_cur in zip(model_shape_shifted, model_shape)]
            biases = [np.zeros((nb_neur,1)) for nb_neur in model_shape]
        else:
            weights = [np.random.rand(nb_neur_prev, nb_neur_cur)
                       for nb_neur_prev, nb_neur_cur in zip(model_shape_shifted, model_shape)]
            biases = [np.random.rand(nb_neur,1) for nb_neur in model_shape]

        values = [np.zeros((1,nb_neur)) for nb_neur in model_shape]
        zs = [np.zeros((nb_neur,1)) for nb_neur in model_shape]
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
        """
        Correct function for neural network. Uses gradient mechanics to upgrade network weights and biases.

        :param inputs: list of network's inputs
            (when single input for network is a list, then it is list of lists)
        :param expected_outputs: list of expected outputs for each network's inputs in previous argument
        :return: 1D list with cost functions values over iterations for all inputs
        """
        # cost need to be minimal
        steps = []  # cost function values over iterations
        for iter in range(MAX_ITERATIONS):
            derivatives_nets = []
            for input, ex_output in zip(inputs, expected_outputs):
                # computate derivatives
                ders_net = Network.create_empty(self.model_shape)
                # computate values for each neuron
                net_output = self.process(input)
                # calculate cost_function for statistics
                steps += [np.sum(np.float_power(net_output - ex_output,2))]
                # C/dvalue from net output
                ex_output = np.matrix(ex_output)
                d_cost_d_a = 2 * (net_output - ex_output)
                for i in reversed(range(1, self.nb_layers)):
                    w, b, _, z = self.layers[i]
                    _, _, v, _ = self.layers[i - 1]

                    d_cost_d_z = np.multiply(np.transpose(d_cost_d_a), sigmoid_der(z))
                    d_cost_d_b = d_cost_d_z
                    d_cost_d_w = np.transpose(v) @ np.transpose(d_cost_d_z)
                    d_cost_d_a = np.transpose(w @ d_cost_d_z)

                    ders_net.layers[i][0] = d_cost_d_w
                    ders_net.layers[i][1] = d_cost_d_b

                # add computed derivatives to array for future average computations
                derivatives_nets += [ders_net]

            # improving net by average derivative from all inputs
            def average(networks_array):
                my_sum = Network.create_empty(networks_array[0].model_shape)
                for network in networks_array:
                    my_sum += network
                return my_sum / len(networks_array)

            me = self
            me -= average(derivatives_nets)

        return steps

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
        return cls(model_shape,empty=True)

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
