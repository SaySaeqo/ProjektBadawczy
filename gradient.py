import numpy as np

from network import Network, average
from organism import *
import numpy
from constants import *

# dodać organizmy i będzie wprosty sposób done
# Może dodać dokładność rozwiązania jako dodatkowy argument
from utils import gradient, sigmoid_der, progress_bar


def gradient_algorithm(func, arg_num, domain_list, min_max, probe_numb):
    """
        :type func: function
    """
    organisms = create_population(arg_num, domain_list, probe_numb)

    iterations = 0
    while (iterations < MAX_ITERATIONS):

        iterations += 1
        # dla każdego organizmu
        for org in organisms:

            # policz gradient
            grad = gradient(func, org.data)  # (derivatives[i](*org.data))
            # zaktualizuj argumenty względem gradientu
            for i in range(arg_num):
                # bezpiecznijsza metoda to przesuwanie o jaką stałą
                # w sensie coś niezależnego bezpośrednio od pochodnej, tak jak teraz
                # ale powinna ona być zmienna od generacji
                # konkretnie maleć, ale o ile? LOGARYTM?!

                org.data[i] -= numpy.sign(grad[i]) * 0.01
        asses(organisms, func)
        organisms = sorted(organisms, key=lambda x: x.ocena, reverse=False)
        print(iterations, '/', MAX_ITERATIONS)

    asses(organisms, func)
    organisms = sorted(organisms, key=lambda x: x.ocena, reverse=False)

    return organisms[0].data


# weź zmień by funkcj gradientowa przyjmowała argumenty:
# funkcja,dziedziny, liczbe argumentów (nawet jeśli miałby nie korzystać), tryb min_max, liczba próbek
# (function,arg_num,domain_list,min_max,probe_num, learn_rate,tol)

# a parametry learn_rate i tol niech będą na końcu (żeby nie trzeba było uwzględniać innych wejść
# wtedy ładnie będzie można także testować tę funkcję
def gradient_func(function, start, min_max=MIN,
                  learn_rate=0.1, max_iter=1_000, tol=0.0001):
    for i in range(max_iter):
        diffs = gradient(function, start)
        diffs = [d if d < learn_rate else learn_rate for d in diffs]
        if all((abs(d) < tol for d in diffs)):
            break
        for i in range(len(start)):
            start[i] += (min_max == MAX) * diffs[i]
            start[i] -= (min_max == MIN) * diffs[i]

    return start


def computate_derivatives(net, input, ex_output):
    """
    Computates derivatives for single pair of output and expected output of net

    :param net: Network to calculate derivatives for
    :param input: single input for net, list of arguments
    :param ex_output: single expected_output, list of arguments
    :return: derivatives packed into Network object
    """
    # var to hold results to return
    ders_net = Network.create_empty(net.model_shape)
    # compute values for each neuron
    net_output = net.process(input)
    # d_cost/d_value from net output
    net_output = np.matrix(net_output)
    ex_output = np.matrix(ex_output)
    d_cost_d_a = 2 * (net_output - ex_output)
    for i in reversed(range(1, net.nb_layers)):
        w, b, _, z = net.layers[i]
        _, _, v, _ = net.layers[i - 1]

        d_cost_d_z = np.multiply(np.transpose(d_cost_d_a), sigmoid_der(z))
        d_cost_d_b = d_cost_d_z
        d_cost_d_w = np.transpose(v) @ np.transpose(d_cost_d_z)
        d_cost_d_a = np.transpose(w @ d_cost_d_z)

        ders_net.layers[i][0] = d_cost_d_w
        ders_net.layers[i][1] = d_cost_d_b

    return ders_net


def net_gradient(net, *args, **kwargs):
    """
    Correct function for neural network. Uses **gradient** mechanics to upgrade network weights and biases.

    :param inputs: list of network's inputs
        (when single input for network is a list, then it is list of lists)
    :param expected_outputs: list of expected outputs for each network's inputs in previous argument
    :return: void
    """

    # cost need to be minimal
    steps = []  # cost function values over iterations

    params = GConst.instance()

    # support for arguments type: list of tuples (input, ex_output)
    if len(args) == 1:
        args = args[0]
    elif len(args) == 2:
        inputs, ex_outputs = args
        args = list(zip(inputs, ex_outputs))

    for iter in range(params.MAX_ITERATIONS):
        batch_count = 0
        semi_step = []

        for batch_ptr in range(0, len(args), params.BATCH_SIZE):
            inout = args[batch_ptr:batch_ptr+params.BATCH_SIZE]

            ders_buffer = []
            for input, ex_output in inout:
                # for debugging
                input = np.matrix(input)
                ex_output = np.matrix(ex_output)
                cost = np.sum(np.float_power(net(input) - ex_output, 2))
                semi_step += [cost]

                # ders is Network object which contains derivatives for weights and biases
                ders = computate_derivatives(net, input, ex_output)
                ders_buffer += [ders]

            if len(ders_buffer) > 0:
                net -= average(ders_buffer)

        if kwargs.get("test_network_simple"):
            steps += semi_step
        else:
            steps += [np.mean(semi_step)]

        if params.MAX_ITERATIONS > 1:
            progress_bar(iter, params.MAX_ITERATIONS)

    return steps
