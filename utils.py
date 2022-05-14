import random

import numpy as np

import testfunctions as tf
from constants import *


def progress_bar(current, max):
    """
    current/max [████████████████____]

    :param current: counter (usually from 0 to max - 1)
    :param max: max value exclusively
    :return: void
    """
    current += 1
    percent = "["
    five_percent = max / 20.0
    current_percent = five_percent
    while current_percent <= max:
        if current >= current_percent:
            percent += "█"
        else:
            percent += " "
        current_percent += five_percent
    percent += "]"
    print(f"\r{current}/{max} {percent}", end="")


def random_data(domain_list):
    """

    :param domain_list:
    :return: list of randomly generated arguments
    """
    data = []
    for i in range(len(domain_list)):
        data.append(random.uniform(domain_list[i][0], domain_list[i][1]))
    return data


def sigmoid(x):
    """

    :param x: real number
    :return: number beetwen 0 and 1
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    """
    Derivative of sigmoid
    :param x:  number
    :return:  number beetwen 0 and 1
    """
    if isinstance(x, np.matrix):
        return np.multiply(sigmoid(x), (1 - sigmoid(x)))
    return sigmoid(x) * (1 - sigmoid(x))


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


class Function:
    def __init__(self, func, domain, min_max=MIN, solutions=None):
        self.function = func
        self.domain = domain
        self.arg_num = len(domain)
        self.min_max = min_max
        self.solutions = solutions

    def random_data(self):
        """

            :param domain_list:
            :return: list of randomly generated arguments
            """
        return random_data(self.domain)

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    @staticmethod
    def get_all() -> list:
        """

        :return: list of all functions to test (Function type elements)
        """
        return [
            Function(
                tf.polynomial,  # 0
                [[-10, 10], [-10, 10]],
                MIN,
                solutions=[[0.55672, 0.55672], [-0.55672, 0.55672],
                           [0.55672, -0.55672], [-0.55672, -0.55672]]
            ), Function(
                tf.rosenbrock,  # 1
                [[-10, 10], [-10, 10]],
                MIN,
                solutions=[[1, 1]]
            ), Function(
                tf.zangwill,  # 2
                [[-150, 150], [-150, 150], [-150, 150]],
                MIN,
                solutions=[[0, 0, 0]]
            ), Function(
                tf.goldenstein,  # 3
                [[-2, 2], [-2, 2]],
                MIN,
                solutions=[[0, -1]]
            ), Function(
                tf.exp_sin,  # 4
                [[-10, 10]],
                MAX,
                solutions=[[0, 1]]
            ), Function(
                tf.himmbelblau,  # 5
                [[-10, 10], [-10, 10]],
                MIN,
                solutions=[[-5, 5]]
            ), Function(
                tf.ackley,  # 6
                [[-30, 30]] * 10,  # N=10
                MIN,
                solutions=[[0] * 10]  # N=10
            ), Function(
                tf.rastrigin,  # 7
                [[-1, 1]] * 10,  # N=10
                MIN,
                solutions=[[0] * 10]  # N=10
            ), Function(
                tf.geem,  # 8
                [[-10, 10], [-10, 10]],
                MIN,
                solutions=[[-0.08984, 0.71266], [0.08984, -0.71266]]
            ), Function(
                tf.sin_sin_exp,  # 9
                [[-10, 10], [-10, 10]],
                MIN
            ), Function(
                tf.sin_sin_exp,  # 10
                [[-10, 10], [-10, 10]],
                MAX
            ), Function(
                tf.lin_exp,  # 11
                [[-10, 10], [-10, 10]],
                MIN,
            ), Function(
                tf.lin_exp,  # 12
                [[-10, 10], [-10, 10]],
                MAX,
            ), Function(
                tf.sin_power6,  # 13
                [[-100000, 100000]],
                MIN,

            ), Function(
                tf.sin_power6,  # 14
                [[-100000, 100000]],
                MAX,
            )]
