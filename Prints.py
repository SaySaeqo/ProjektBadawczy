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