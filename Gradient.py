import copy

import numpy as nmp
from Network import network
from Functions import sigmoid

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


def sigmoid_derivative(x):
    '''pochodna sigmoid'''
    result= sigmoid(x)*(1-sigmoid(x))
    return result

def calc_z_L(layer_index,neuron_index,net:network):
    '''funkcja pomocnicza wyliczająca z_L, czyli argument funkcji sigmoid, wynikający z
    sumy aktywacji poprzenich neuronów, przemnożonych przez odpowiednie wagi'''
    z_L=0

    for i in range(net.nrn_nmb[layer_index-1]):
        z_L += net.matrices[layer_index][neuron_index][i] * net.neuron_values[layer_index - 1][i]
    z_L+= net.biases[layer_index][neuron_index]

    return z_L
def cost(result,expected):
    '''w sumie nie używane XD, ale do genetycznego może się przydać'''
    sum=0;
    for i in range(len(result)):
        sum+=(result[i]-expected[i])**2
    return sum
def net_derivative(layer_index,neruon_index,net:network, expected_result, weight_index = 0, modify_weight = False, modify_bias= False, calc_neuron = False):
    '''funkcja licząca pochodną po wybranym elemencie:
    -wadze
    -biasie
    -wartości aktywacji nueuronu'''
    #liczenie pochodnej po wadze
    if modify_weight:
        #z_L=w_L * a _(L-1) + b_L
        z_L=calc_z_L(layer_index,neruon_index,net)

        #grad= a_(L-1) * sigmoid' (z_L)
        grad = net.neuron_values[layer_index-1][weight_index]*sigmoid_derivative(z_L)

        #grad *= dC/da_(L)
        grad*=net_derivative(layer_index,neruon_index,net,expected_result,calc_neuron=True)

    # po biasie
    elif modify_bias:
        #liczenie z_L jak wyżej
        z_L = calc_z_L(layer_index, neruon_index, net)
        grad=sigmoid_derivative(z_L) * net_derivative(layer_index,neruon_index,net,expected_result,calc_neuron=True)

    # a tu po wartości aktywacji neuronu
    elif calc_neuron:
        if layer_index == net.layers-1:
            #to prosta pochodna po Co
            grad= 2*(net.neuron_values[layer_index][neruon_index] - expected_result[neruon_index])
        else:
            #rekurencja
            grad=0
            for i in range(net.nrn_nmb[layer_index+1]):
                #z_(L+1) tutaj to jest tak właściwie, ale nie da się plusa napisać w nazwie zmiennej
                z_L = calc_z_L(layer_index+1, i, net)
                # suma : w_jk^(L+1) + sigmoid'(z_j^(L+1) * dc/daj^(L+1)
                grad+=net.matrices[layer_index+1][i][neruon_index] * sigmoid_derivative(z_L) * net_derivative(layer_index+1,i,net,expected_result,calc_neuron=True)

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

    #dla kazsdej warstwy
    for i in range(1,net.layers):
        #dla każdego neuronu z tej warstwy
        for j in range(net.nrn_nmb[i]):
            #policz wagi od neuronów z lewej warstwy
            for k in range(net.nrn_nmb[i-1]):
                matrices_der[i][j][k] = net_derivative(i,j,net,expected_result,weight_index=k,modify_weight=True)
            #policz bias dla neuronu
            biases_der[i][j] = net_derivative(i, j, net, expected_result, modify_bias=True)
    return matrices_der,biases_der