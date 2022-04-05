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
        func.domain = [[-2, 2], [-2, 12]]
        func.arg_num = 2
        func.min_max = MIN
        func.solutions=[[0,-1]]
    return func
