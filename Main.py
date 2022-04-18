from Tests import  *


#test_algorithms()

net=network(4,[1,1,1,1],net_gradient)

print (net)

print(net.process([1]))
net.correct([0])
