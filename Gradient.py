import copy

import numpy as nmp
from Network import network

def derivative(func,args,n):
    h=0.000001
    new_args=args.copy()
    new_args[n]+=h
    return (func(new_args)-func(args))/h




def gradient(func,args_values):
    result=[];
    for i in range(len(args_values)):
        result.append(derivative(func,args_values,i))
    return result


def sigmoid(x):
    return 1/(1+nmp.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

def cost(result,expected):
    sum=0;
    for i in range(len(result)):
        sum+=(result[i]-expected[i])**2
    return sum
def net_derivative(layer_index,neruon_index,net:network, expected_vresult, modify_weight = False, modify_bias= False):
    if modify_weight:
        #z_L=w_L * a _(L-1) + b_L
        z_L=net.matrices[layer_index][0][0]*net.neuron_values[layer_index-1][0]+net.biases[layer_index][0]

        #grad= a_(L-1) * sigmoid' (z_L)
        grad = net.neuron_values[layer_index-1][0]*sigmoid_derivative(z_L)
        layer_index+=1
        while(layer_index<net.layers):
            z_L = net.matrices[layer_index][0][0] * net.neuron_values[layer_index - 1][0] + net.biases[layer_index][0]
            grad *= net.neuron_values[layer_index - 1][0] * sigmoid_derivative(z_L)
            layer_index+=1
        grad*=2*(net.neuron_values[layer_index-1][0]-expected_vresult[0])
    elif modify_bias:
        pass
    return grad

def net_gradient(neuron_values,expected_result,net:network):
    '''
    :param neuron_values: tablica wektorów. w każdym wektorze sa wartości aktywacji poszczególnych neuronów
    :param layers:
    :param matrices:
    :param biases:
    :return:
    '''
    matrices_der = copy.deepcopy(net.matrices)
    biases_der = copy.deepcopy(net.biases)

    #POPRAWIC DLA DOWOLNEJ SIECI XD
    for i in range(1,net.layers):
        matrices_der[i][0] = net_derivative(i,0,net,expected_result,modify_weight=True)

    return matrices_der,biases_der