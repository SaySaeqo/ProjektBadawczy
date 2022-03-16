# Funkcje,zmienne Snake-case wszystko male i podlogi
# klasy - camelCase 
# stale - krzykliwy tabulator

import random

MIN = 0
MAX = 1
PROBE_NUMBER=10


class organism(object):		
	
	def __init__(self,arg_num):		
		data=[None]*arg_num
		
class domain():
	def __init__(self):
		beg=0
		end=0


def create_population(org,arg_num,domain_list,probe_num):
	for i in range(probe_num):		
		new_organism=organism(arg_num)
		temp_list=[None]*arg_num
		#generowanie losowych wartości początkowych
		for j in range(arg_num):
			#dlaczego linia niżej nie działa? Z tego powodu zrobiłem przez listę nową (temp) :/						
			#new_organism.data[j]=random.randrange(domain_list[j][0],2)						
			temp_list[j]=random.uniform(domain_list[j][0],domain_list[j][1])
			new_organism.data=temp_list
		org.append(new_organism)				
	return org

def asses(organisms,function):
	for org in organisms:
		org.ocena=function(org.data)

#śmieć na razie ale zostawiam na wypadek
def reproduction(old_generation,arg_num,domain_list,min_max,probe_num):
	#liczenie sumy
	suma=0
	for org in old_generation:
		suma+=org.ocena
		
	new_generation=[]
	org_counter=0
	if min_max == MAX:		
		for org in old_generation:
			for i in range(int(org.ocena/suma*probe_num)):
				org_counter+=1
				new_generation.append(org)
		
		if org_counter<probe_num:
			for i in range(org_counter,probe_num):
				org=organism(arg_num)
				org.data=old_generation[0].data
				new_generation.append(org)
				
	elif min_max == MIN:		
		
		for org in old_generation:		
			for i in range(int(pow(1-org.ocena/suma,-1)*probe_num)):				
				org_counter+=1
				new_generation.append(org)
				
		if org_counter<probe_num:
			for i in range(org_counter,probe_num):
				org=organism(arg_num)				
				org.data=old_generation[0].data
				new_generation.append(org)	
	return new_generation


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
	
	
	
	#zwraca liste zawierającą średnie po każdej pozycji
def average(list_a, list_b):
	output=[None]*len(list_a)
	for i in range(len(list_a)):
		output[i]=(list_a[i]+list_b[i])/2
	return output
	
def cross_breeds(generacja,probe_num):
	copy=generacja.copy()
	children=[]
	for i in range(int(probe_num/2)):		
		#para bierze losowe 2 organizmy z listy
		parents=random.sample(copy, 2)		
		copy.remove(parents[0])
		copy.remove(parents[1])		
		data_a=parents[0].data
		data_b=parents[1].data		
		avg=average(data_a,data_b)		
		child=organism(0)		
		child.data=avg	
		children.append(child)						
	return children
	
	
def randomData(domain_list,arg_num):
	data=[None]*arg_num
	for i in range(arg_num):
		data[i]=(random.uniform(domain_list[i][0],domain_list[i][1]))
	return data	
	
def mutate(generation,domain_list,arg_num):
	mutated=[]
	for org in generation:
		if random.randint(0,100)>95 :
			org.data=randomData(domain_list,arg_num)
	
def select_best(parents, children,min_max):
	organisms=parents.copy()+children.copy()
	if min_max == MAX:
		organisms=sorted(organisms, key=lambda x: x.ocena, reverse=True)
	elif min_max == MIN:
		organisms=sorted(organisms, key=lambda x: x.ocena, reverse=False)		
			
	organisms=organisms[:len(organisms)-len(children)]
	return organisms

def genetic_func(function,arg_num,domain_list,min_max,probe_num):
	organisms=[]

	generation_counter=0
	create_population(organisms,arg_num,domain_list,probe_num)
	print_generation(organisms,generation_counter,function)
	asses(organisms,function)
	

	#while not(Warunek(organisms)):
	while not(generation_counter == 1000):
		generation_counter+=1	
		#wcześniejsza próba:
		# organisms = reproduction(organisms,arg_num,domain_list,min_max,probe_num)
		# organisms=cross_breeds(organisms)
		# mutate(organisms,domain_list,arg_num)
		
		#lepsza wersja: (bardziej rzeczywista?)
		#krzyżyj organizmy(twórz ich dzieci)
		children=cross_breeds(organisms,probe_num)		
		#mutj dzieci
		mutate(children,domain_list,arg_num)	
		#ocen dzieci
		asses(children,function)			
		#wybierz teraz najlepiej przystosowane		
		organisms=select_best(organisms, children,min_max)		
		
		print_generation(organisms,generation_counter,function)		
			
	#ustawienie wyniku w zależności czego szukaliśmy		
	if min_max == MAX:
		organisms=sorted(organisms, key=lambda x: x.ocena, reverse=True)
	elif min_max == MIN:
		organisms=sorted(organisms, key=lambda x: x.ocena, reverse=False)
	return organisms[0].data
	
#przyklad wywolania	
#print("wynik",genetic_func(get_function(1),2,[[-10,10],[-10,10]],MIN,PROBE_NUMBER))


# to do
#    mutacje na liczba rzeczywistych,
#


