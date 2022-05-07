from functools import reduce

from Constants import *
from utils import sigmoid
from utils import sigmoid_der
from Network import Network, Neuron
import numpy as np


def derivative(func, args, n):
    """
    Simple derivative by definition

    :param func: function which take args as arguments
    :param args: list of numbers
    :param n: index of argument from args derivatise is about
    :return: number
    """
    h = 0.000001
    new_args = args.copy()
    new_args[n] += h
    return (func(new_args) - func(args)) / h


def gradient(func, args_values):
    """
    Simple gradient for traditional multiarguments functions

    :param func: function which take args_values as arguments
    :param args_values: list of numbers
    :return: vector of derivatives (numbers)
    """
    result = []
    for i in range(len(args_values)):
        result.append(derivative(func, args_values, i))
    return result


def cost_func(output, expected_output):
    """
    Square cost function for net gradient function

    :param output: list of net's outputs
    :param expected_output: list of target outputs
    :return: number
    """
    return sum(((a - b) ** 2 for a, b in zip(output, expected_output)))


def cost_func_der(neuron_output, neuron_expected_output):
    """
    Derivative of square cost function for net gradient function

    :param neuron_output: single neuron output
    :param neuron_expected_output: single target output
    :return: number
    """
    return 2 * (neuron_output - neuron_expected_output)


def net_gradient(net: Network, inputs, expected_outputs):
    """
    Correct function for neural network. Uses gradient mechanics to upgrade network weights and biases.

    :param net: Network
    :param inputs: list of network's inputs
        (when single input for network is a list, then it is list of lists)
    :param expected_outputs: list of expected outputs for each network's inputs in previous argument
    :return: void
    """
    # cost need to be minimal

    for _ in range(MAX_ITERATIONS):
        derivatives_nets = []
        for input, ex_output in zip(inputs, expected_outputs):

            # computate values for each neuron
            net.process(input)

            # computate derivatives
            ders_net = Network.create_empty(net.model_shape)
            # C/dvalue from net output
            net_output = (neuron.value for neuron in net.layers[-1])
            dvalues = (cost_func_der(cur, ex) for cur, ex in zip(net_output, ex_output))
            for i in reversed(range(1, len(net.model_shape))):

                # new C/dvalue for each layer (array of C/dvalue for each neuron)
                new_dvalues = [0] * net.model_shape[i - 1]

                for neuron, dvalue in zip(net.layers[i], dvalues):

                    # tmp = C/dvalue * dvalue/dz
                    tmp = dvalue * sigmoid_der(neuron.z)

                    # add to structure for derivatives
                    derivative_neuron = Neuron()
                    derivative_neuron.bias = 1 * tmp
                    derivative_neuron.weights = [prev.value * tmp for prev in net.layers[i - 1]]
                    ders_net.layers[i] += [derivative_neuron]

                    # sum for new C/dvalue from next layer's neurons and previous C/dvalue
                    sums_addons = [weight * tmp for weight in neuron.weights]
                    new_dvalues = \
                        [base + a for base, a in zip(new_dvalues, sums_addons)]

                # switch to new C/dvalue
                dvalues = new_dvalues

            # add computed derivatives to array for future average computations
            derivatives_nets += [ders_net]

        # improving net by average derivative from all inputs
        def average(networks_array):
            sum = networks_array[0]
            first = True
            for network in networks_array:
                if first:
                    first = False
                    continue
                sum += network
            return sum / len(networks_array)
        net += average(derivatives_nets)
