def polynomial(args):
  return pow(args[0],4)+pow(args[1],4)-0.62*pow(args[0],2)-0.62*pow(args[1],2)

def rosenbrock(args):
	return 100*pow(args[1]-pow(args[0],2),2)+pow(1-args[0],2)

def get_function(n):
	if n==1:
		return polynomial
	if n==2:
		return rosenbrock
