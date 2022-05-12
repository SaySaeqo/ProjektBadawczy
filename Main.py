from decimal import Decimal

from Tests import *
from re import sub


# test_algorithms()

# test_neuron()
# test_neuron_libs("neurolab")
# test_neuron_libs("pytorch")
# oczekiwane wyniki:
# 1 1 1 1
# 1 0 1 1
# 1 1 0 0
# 0 0 0 0
#PlotIris("iris_single_try")
#PlotIris("iris_genetic_learn_many_attempts")
#PlotIris("iris_genetic_time_comparison")
#PlotIris("test_gradient")
#PlotIris("test_genetic")
#PlotIris("test_generalism")
#PlotBeans("beans_single_try")
#PlotBeans("beans_genetic_learn_many_attempts")
#PlotBeans("test_gradient")
#PlotBeans("test_generalism")
PlotRisin("risin_single_try")


''' # tu test na zerach i jedynkach

net = genetic_Network([4, 3, 2], 50)
net.start_learning()
for i in range(100):
    net.process([1, 0, 0, 1])
    net.correct([1, 1])
    net.process([0, 1, 1, 0])
    net.correct([1, 0])
    net.process([1, 1, 1, 1])
    net.correct([0, 1])
    net.process([1, 0, 1, 0])
    net.correct([0, 0])
    # net.print_assesments()
    # print('-----------------------------')

net.stop_learning()
print(net.process([1, 0, 0, 1])) # 1 1 
print(net.process([0, 1, 1, 0])) # 1 0
print(net.process([1, 1, 1, 1])) # 0 1
print(net.process([1, 0, 1, 0])) # 0 0

'''


#\/ do usuniecia \/

'''
net=network([4,3,3,3],net_gradient,1)

net.start_learning()
n=2
for i in range(2000):
    net.process([5.5/n,2.4/n,3.7/n,1.0/n])
    net.correct([1, 0, 0])
    net.process([4.4/n, 3.2/n, 1.3/n, 0.2/n])
    net.correct([0, 1, 0])
    net.process([5.9/n,3.0/n,5.1/n,1.8/n])
    net.correct([0, 0, 1])

    print(i)
net.stop_learning()
print(net.process([5.5/n,2.4/n,3.7/n,1.0/n]))
print(net.process([4.4/n,3.2/n,1.3/n,0.2/n]))
print(net.process([5.9/n,3.0/n,5.1/n,1.8/n]))
#'''