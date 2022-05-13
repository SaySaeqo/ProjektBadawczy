import math
import random

from utils import random_data
from utils import gradient
from constants import *


def mutate_new_random(generation, domain_list, arg_num):
    mutated = []
    for org in generation:
        if random.randint(0, 100) > 95:
            org.data = random_data(domain_list, arg_num)


def mutate_change_little(generation, domain_list, arg_num):
    mutated = []
    for org in generation:
        for i in range(len(org.data)):
            if random.randint(0, 100) > 95:
                delta = 0.01
                if random.randint(0, 1) == 1:
                    org.data[i] += delta
                else:
                    org.data[i] -= delta


def mutate_gradient_wise(generation,func, domain_list,min_max):
    mutated = []
    for org in generation:
        if random.randint(0, 100) > 95:
            grad = gradient(func, org.data)
            for i in range(len(org.data)):
                if(min_max == MIN):
                    org.data[i] -= grad[i]
                elif min_max == MAX:
                    org.data[i] += grad[i]

# def mutateBits(data,number):
