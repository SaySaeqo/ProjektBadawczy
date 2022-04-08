from Tests import  *
N=1000

#test_algorithms()

nrn_nmb=[3,5,5,4]
net=network(nrn_nmb,net_gradient)

#print (net)
for i in range(N):
    net.process([ 0, 0, 0])
    net.correct([1,1,1,1])
    net.process([0, 1, 0])
    net.correct([1, 0, 1, 1])
    net.process([0, 1, 1])
    net.correct([1, 1, 0, 0])
    net.process([1, 1, 1])
    net.correct([0,0,0, 0])
    print(i,'/',N)
#print(net)
#                          oczekiwane wyniki:
print(net.process([0,0,0])) # 1 1 1 1
print(net.process([0,1,0])) # 1 0 1 1
print(net.process([0,1,1])) # 1 1 0 0
print(net.process([1,1,1])) # 0 0 0 0