def polynomial(x,y):
  return pow(x,4)+pow(y,4)-0.62*pow(x,2)-0.62*pow(y,2)

def rosenbrock(x,y):
	return 100*pow(y-pow(x,2),2)+pow(1-x,2)

def get_function(n):
	if n==1:
		return polynomial
	if n==2:
		return rosenbrock
