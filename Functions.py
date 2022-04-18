import math
from Function import *

def polynomial(args):
  return pow(args[0],4)+pow(args[1],4)-0.62*pow(args[0],2)-0.62*pow(args[1],2)

def rosenbrock(args):
	return 100*pow(args[1]-pow(args[0],2),2)+pow(1-args[0],2)

def zangwill(args):
    ingredients = []
    ingredients.append(args[0]-args[1]+args[2])
    ingredients.append(-args[0]+args[1]+args[2])
    ingredients.append(args[0]+args[1]-args[2])
    ingredients = map(lambda a : a*a, ingredients)
    return sum(ingredients)

def goldenstein(args):
    x = args[0]
    y = args[1]
    factors = []
    factors.append(1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))
    factors.append(30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))
    return factors[0]*factors[1]

def exp_sin(args):
    x=args[0]
    sum_a= math.exp(-2*math.log10(2) * (x-0.08)**2 / 0.854**2)
    sum_b= (math.sin(5*math.pi * (abs(x)**(3/4) - 0.05)))**6
    return sum_a * sum_b

def himmbelblau(args):
    x=args[0]
    y=args[1]
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2 - 200

def ackley(args):
    sum1=0
    sum2=0
    n=len(args)

    #liczenie sum
    for i in range(n):
        sum1+=args[i]**2
        sum2+=math.cos(2*math.pi*args[i])

    # a teraz argumenty exponetów
    a=-0.2*math.sqrt(1/n)*sum1
    b=1/n * sum2
    return -20 * math.exp(a) - math.exp(b)

def rastrigin(args):
    sum = 0
    n = len(args)

    # liczenie sum
    for i in range(n):
        sum += args[i] ** 2 - math.cos(18*args[i])
    return sum

def geem(args):
    x=args[0]
    y=args[1]

    return 4 * x**2 - 2.1 * x**4 +1/3 * x**6 + x*y - 4*y**2 + 4*y**4

def sin_sin_exp(args):
    x=args[0]
    y=args[1]

    return math.sin(x)*math.sin(y)*math.exp(-x**2-y**2)

def lin_exp(args):
    x = args[0]
    y = args[1]
    return x*math.exp(-x**2-y**2)

def sin_power6(args):
    x=args[0]
    return (math.sin(5.1 * math.pi * x + 0.5))**6


def get_function(n):
    '''
    
    :param n: indeks funkcji 
    :return: obiekt Function
    '''
    func=Function()
    if n==1:
        func.function = polynomial
        func.domain=[[-10,10],[-10,10]]
        func.arg_num=2
        func.min_max=MIN
        func.solutions=[[0.55672,0.55672],[-0.55672,0.55672],[0.55672,-0.55672],[-0.55672,-0.55672]]
    if n==2:
        func.function = rosenbrock
        func.domain = [[-10, 10], [-10, 10]]
        func.arg_num = 2
        func.min_max = MIN
        func.solutions=[[1,1]]
    if n==3:
        func.function = zangwill
        func.domain = [[-150, 150], [-150, 150],[-150,150]]
        func.arg_num = 3
        func.min_max = MIN
        func.solutions=[[0,0,0]]
    if n==4:
        func.function = goldenstein
        func.domain = [[-2, 2], [-2, 2]]
        func.arg_num = 2
        func.min_max = MIN
        func.solutions=[[0,-1]]
    if n==5:
        func.function = exp_sin
        func.domain = [[-10, 10]]
        func.arg_num = 1
        func.min_max = MAX
        func.solutions=[[0, 1]]
    if n==6:
        func.function = himmbelblau
        func.domain = [[-10, 10],[-10,10]]
        func.arg_num = 2
        func.min_max = MIN
        func.solutions=[[-5, 5]]
    if n==7:
        func.function = ackley
        n=10
        func.domain = [[-30, 30]]*n
        func.arg_num = n
        func.min_max = MIN
        func.solutions=[[0]*n]
    if n==8:
        func.function = rastrigin
        n = 10
        func.domain = [[-1, 1]]*n
        func.arg_num = n
        func.min_max = MIN
        func.solutions=[[0]*n]
    if n==9:
        func.function = geem
        func.domain = [[-10, 10], [-10, 10]]
        func.arg_num = 2
        func.min_max = MIN
        func.solutions = [[-0.08984, 0.71266],[0.08984,-0.71266]]
    if n==10:
        func.function = sin_sin_exp
        func.domain = [[-10, 10], [-10, 10]]
        func.arg_num = 2
        func.min_max = MIN
        func.solutions = [[-0,0],[0,0]] #NIEZNANE
    if n==11:
        func.function = sin_sin_exp
        func.domain = [[-10, 10], [-10, 10]]
        func.arg_num = 2
        func.min_max = MAX
        func.solutions = [[-0,0],[0,0]] #NIEZNANE
    if n==12:
        func.function = lin_exp
        func.domain = [[-10, 10], [-10, 10]]
        func.arg_num = 2
        func.min_max = MIN
        func.solutions = [[-0,0],[0,0]] #NIEZNANE
    if n==13:
        func.function = lin_exp
        func.domain = [[-10, 10], [-10, 10]]
        func.arg_num = 2
        func.min_max = MAX
        func.solutions = [[-0,0],[0,0]] #NIEZNANE
    if n==14:
        func.function = sin_power6
        func.domain = [[-100000, 100000]] #duży przedział
        func.arg_num = 1
        func.min_max = MIN
        func.solutions = [[-0,0],[0,0]] #skonćżenie wiele
    if n==15:
        func.function = sin_power6
        func.domain = [[-100000, 100000]] #duży przedział
        func.arg_num = 1
        func.min_max = MAX
        func.solutions = [[-0,0],[0,0]] #skonćżenie wiele
    return func