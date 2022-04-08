import copy

from Gradient import *
import random
from Organism import *
import numpy
import math
import Constants


# dodać organizmy i będzie wprosty sposób done
# Może dodać dokładność rozwiązania jako dodatkowy argument

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
def gradient_func(function, start, min_max=Constants.MIN, \
                  learn_rate=0.001, max_iter=1_000, tol=0.0001):
    for i in range(max_iter):
        diff = gradient(function, start)
        diff = list(map(lambda a: a * learn_rate, diff))
        if all(map(lambda a: abs(a) < tol, diff)):
            break
        for i in range(len(start)):
            start[i] += (min_max == Constants.MAX) * diff[i]
            start[i] -= (min_max == Constants.MIN) * diff[i]

    return start
