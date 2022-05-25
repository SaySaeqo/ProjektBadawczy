from singleton.singleton import Singleton

MIN = 0
MAX = 1
PROBE_NUMBER = 100

DELTA_MUTATION = 0.01
ALGORITHMS_NUMBER = 4

MODIFY_WEIGHT = 0
MODIFY_BIAS = 1
CALC_NEURON = 2

INPUT = 0
OUTPUT = 1
MAX_GENERATIONS = MAX_ITERATIONS = 200  # ?? nie używam już


@Singleton
class GeneticConst:
    MAX_GENERATIONS = 150
    POPULATION_SIZE = 30
    MUTATION_CHANCE = 0.2
    MUTATION_RATE = 0.5
    BATCH_SIZE = 50


@Singleton
class GradientConst:
    MAX_ITERATIONS = 200
    BATCH_SIZE = 10
