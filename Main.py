from decimal import Decimal

from Tests import *
from re import sub
import csv

# test_algorithms()

# test_neuron()
# test_neuron_libs("neurolab")
# test_neuron_libs("pytorch")
# oczekiwane wyniki:
# 1 1 1 1
# 1 0 1 1
# 1 1 0 0
# 0 0 0 0


def name_to_tuple(name):
    if name=="Iris-setosa":
        return [1, 0, 0]
    elif name=="Iris-versicolor":
        return [0, 1, 0]
    elif name=="Iris-virginica":
        return [0,0,1]

net = geneticNetwork([4, 3, 3, 3], 50)
net_normal= network([4,3,3,3],net_gradient)
file = open("iris.data")
csvreader = csv.reader(file)

net.start_learning()

table=[]
for row in csvreader:
    table.append(row)

random.shuffle(table)
for i in range(147):
    row= table[i]
    name=row[4]
    row=row[:4]
    for i in range(4):
        row[i]= float(row[i])
    expected=name_to_tuple(name)
    #print (row,expected)
    net.process(row)
    net.correct(expected)
    net_normal.process(row)
    net_normal.correct(expected)
net.stop_learning()

for i in range(147,150):
    row= table[i]
    name=row[4]
    row=row[:4]
    for i in range(4):
        row[i]= float(row[i])
    expected=name_to_tuple(name)
    #print (row,expected)
    print(expected,net_normal.process(row))
    print(expected,net.process(row))

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