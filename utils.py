import random

import numpy as nmp

import testfunctions as tf
from Constants import *


def random_data(domain_list):
    data = []
    for i in range(len(domain_list)):
        data.append(random.uniform(domain_list[i][0], domain_list[i][1]))
    return data


def sigmoid(x):
    """funkcja zwracają liczbę od 0 do 1 dla dowolnej liczby rzeczywistej"""
    return 1 / (1 + nmp.exp(-x))


class Function:
    def __init__(self, func, domain, min_max=MIN, solutions=None):
        self.function = func
        self.domain = domain
        self.arg_num = len(domain)
        self.min_max = min_max
        self.solutions = solutions

    def random_data(self):
        return random_data(self.domain, self.arg_num)

    @staticmethod
    def get_all() -> list:
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
