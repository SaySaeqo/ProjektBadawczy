
def print_generation(generation,n,function):
	print("generation:",n)	
	for org in generation:
		print(org.data, function(org.data))


#funkcja debuggerowa, bo wartości się powtarzały
#to coś zaczęło mi nie pasować. Źle kopiowałem obiekty, ale to już fixed
def print_generation_addres(generation,n):
	print("generation:",n)	
	for org in generation:
		print(org)
	
	
## testowanie gradientu, learn_rate hardcodowany
for i in range(1,5):
    lowest = math.inf
    it = math.inf
    ar = 0
    for _ in range(100):
        func, start = Functions.get_function(i)
        lr = 0.00003
        if i ==3:
            lr = 150.0/300000.0
        args = gradient_func(func, start,
                            learn_rate=lr, tol=0.0000001, max_iter=10_000)
        fx = func(args)
        if lowest > fx:
            lowest = fx
            ar = args
    print(i, ar, lowest)
