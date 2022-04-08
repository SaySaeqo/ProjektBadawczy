from Constants import *
import random


def random_data(domain_list, arg_num):
    data = [None] * arg_num
    for i in range(arg_num):
        data[i] = (random.uniform(domain_list[i][0], domain_list[i][1]))
    return data


class Function:
    def __init__(self):
        self.domain=[]
        self.function=0
        self.arg_num=0
        self.min_max=MIN
        self.solutions=[]
        #urzyć czy nie użyć?
        #o to jest pytanie? XD

    def random_data(domain_list, arg_num):
        return random_data(domain_list,arg_num)
