import copy

from Network import *

class mixNetwork(geneticNetwork):
    def __init__(self, nrn_nmb,correct_function, net_numb, gen_batch_size,grad_batch_size,allInputs,allExpected,geneticWaitIterations ,learnBase = 10 , fractionLearnRate= 3/4, learnSuppresion = 1):
        self.genetic = True
        self.learning = False
        self.net_numb = net_numb
        self.correct_function = correct_function
        self.nets = [None] * net_numb
        self.args = []
        self.nrn_nmb = nrn_nmb
        self.expected_result = []
        self.last_args = []
        self.last_expected = []
        self.batches = (gen_batch_size,grad_batch_size)
        self.batch_size = gen_batch_size
        self.learnBase = learnBase
        self.fractionLearnRate = fractionLearnRate
        self.learnCounter = 1
        self.learnSuppresion = learnSuppresion

        self.allInputs = allInputs
        self.allExpected = allExpected
        self.bestAccuracy = None
        self.geneticMaxWaitIterations = geneticWaitIterations
        self.geneticIterations = 0
        self.bestNet = network(nrn_nmb,correct_function, gen_batch_size)

        for i in range(net_numb):
            self.nets[i] = network(nrn_nmb, correct_function, gen_batch_size,learnBase = learnBase ,fractionLearnRate=fractionLearnRate, learnSuppresion = learnSuppresion)


    def checkProgress(self):
        total_error = self.nets[0].total_error(self.allInputs, self.allExpected)
        return total_error

    def correct(self, expected_result):
        if self.genetic:
            #siec czuje sie genetyczna
            geneticNetwork.correct(self,expected_result)

            totalError = self.checkProgress()

            if self.bestAccuracy == None or totalError < self.bestAccuracy:
                self.bestAccuracy = totalError
                # zapisanie stanu sieci
                self.bestNet = copy.deepcopy(self.nets[0])
            else:
                self.geneticIterations += 1

                if (self.geneticIterations > self.geneticMaxWaitIterations):
                    self.genetic = False
                    # przywocenie najlepszej sieci
                    self.bestNet.batch_size = self.batches[1] #rozmiar batchu radientowego
                    self.bestNet.learnCounter = 0
                    self.nets[0] = self.bestNet


        else:
            #siec jest czuje sie FF
            network.correct(self.nets[0],expected_result)


    def start_learning(self):
        '''bedzie to mówiło, że podczas process, '''
        for net in self.nets:
            net.start_learning()

    def stop_learning(self):
        for net in self.nets:
            net.stop_learning()