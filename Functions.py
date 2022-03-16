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
	if n==1:
		return polynomial
	if n==2:
		return rosenbrock
	if n==3:
		return zangwill
	if n==4:
		return goldenstein
