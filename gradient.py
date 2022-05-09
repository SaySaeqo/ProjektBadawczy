import numpy as np

from network import Network
from organism import *
import numpy
from constants import *


# dodać organizmy i będzie wprosty sposób done
# Może dodać dokładność rozwiązania jako dodatkowy argument
from utils import gradient, sigmoid_der


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


def net_gradient(net, inputs, expected_outputs):
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
            ders_net = Network.create_empty(net.model_shape)
            # computate values for each neuron
            net_output = net.process(input)
            # calculate cost_function for statistics
            steps += [np.sum(np.float_power(net_output - ex_output, 2))]
            # C/dvalue from net output
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

            # add computed derivatives to array for future average computations
            derivatives_nets += [ders_net]

        # improving net by average derivative from all inputs
        def average(networks_array):
            my_sum = Network.create_empty(networks_array[0].model_shape)
            for network in networks_array:
                my_sum += network
            return my_sum / len(networks_array)

        net -= average(derivatives_nets)

    return steps