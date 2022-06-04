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
MAX_GENERATIONS = MAX_EPOCHS = 200  # ?? nie używam już


@Singleton
class GeneticConst:
    MAX_GENERATIONS = 150
    POPULATION_SIZE = 30
    MUTATION_CHANCE = 0.2
    MUTATION_RATE = 0.5
    BATCH_SIZE = 50
    SHOW_PROGRESS = False

    def __repr__(self):
        return f"MAX GENARATIONS: {self.MAX_GENERATIONS}\n" \
               f"POPULATION SIZE: {self.POPULATION_SIZE}\n" \
               f"MUTATION CHANCE: {self.MUTATION_CHANCE}\n" \
               f"MUTATION RATE: {self.MUTATION_RATE}\n" \
               f"BATCH_SIZE: {self.BATCH_SIZE}"


@Singleton
class GradientConst:
    MAX_EPOCHS = 200
    BATCH_SIZE = 10
    SHOW_PROGRESS = False

    def __repr__(self):
        return f"MAX EPOCHS: {self.MAX_EPOCHS}\n" \
               f"BATCH_SIZE: {self.BATCH_SIZE}"