
from Organism import *
from Prints import *
from Breed_Crosses import *
import random
from Mutations import *


#nazwa funkcji
#genetyczna_MetodaTworzeniaDzieci_RodzajMutacji

def genetic_func_mean_random(function,arg_num,domain_list,min_max,probe_num):

	generation_counter = 0
	MAX_GENERATION = 10000
	organisms=create_population(arg_num,domain_list,probe_num)
	print_generation(organisms,generation_counter,function)
	asses(organisms,function)
	

	#while not(Warunek(organisms)):
	while not(generation_counter == MAX_GENERATION):
		generation_counter+=1	
				
		#krzyżyj organizmy(twórz ich dzieci)
		children=cross_breeds_avg(organisms,probe_num)		
		#mutuj dzieci
		mutate_new_random(children,domain_list,arg_num)	
		#ocen dzieci
		asses(children,function)			
		#wybierz teraz najlepiej przystosowane		
		organisms=select_best(organisms, children,min_max)		

		print(generation_counter, '/', MAX_GENERATION)
		#print_generation(organisms,generation_counter,function)
			
	#ustawienie wyniku w zależności czego szukaliśmy		
	if min_max == MAX:
		organisms=sorted(organisms, key=lambda x: x.ocena, reverse=True)
	elif min_max == MIN:
		organisms=sorted(organisms, key=lambda x: x.ocena, reverse=False)
	return organisms[0].data
	
	
	
def genetic_func_mean_change(function,arg_num,domain_list,min_max,probe_num):

	generation_counter=0
	MAX_GENERATION=10000
	organisms=create_population(arg_num,domain_list,probe_num)
	print_generation(organisms,generation_counter,function)
	asses(organisms,function)
	

	#while not(Warunek(organisms)):
	while not(generation_counter == MAX_GENERATION):
		generation_counter+=1	
				
		#krzyżyj organizmy(twórz ich dzieci)
		children=cross_breeds_avg(organisms,probe_num)		
		#mutuj dzieci
		mutate_change_little(children,domain_list,arg_num)	
		#ocen dzieci
		asses(children,function)			
		#wybierz teraz najlepiej przystosowane		
		organisms=select_best(organisms, children,min_max)		

		print(generation_counter, '/', MAX_GENERATION)
		#print_generation(organisms,generation_counter,function)
			
	#ustawienie wyniku w zależności czego szukaliśmy		
	asses(organisms,function)
	if min_max == MAX:
		organisms=sorted(organisms, key=lambda x: x.ocena, reverse=True)
	elif min_max == MIN:
		organisms=sorted(organisms, key=lambda x: x.ocena, reverse=False)
	return organisms[0].data

def genetic_func_mean_gradient_change_litle(function,arg_num,domain_list,min_max,probe_num):
	generation_counter = 0
	MAX_GENERATION = 10000
	organisms = create_population(arg_num, domain_list, probe_num)
	print_generation(organisms, generation_counter, function)
	asses(organisms, function)

	# while not(Warunek(organisms)):
	while not (generation_counter == MAX_GENERATION):
		generation_counter += 1

		# krzyżyj organizmy(twórz ich dzieci)
		children = cross_breeds_avg(organisms, probe_num)
		# mutuj dzieci
		mutate_gradient_wise(children, domain_list,function,min_max)
		# ocen dzieci
		asses(children, function)
		# wybierz teraz najlepiej przystosowane
		organisms = select_tournament(organisms, children, min_max)

		print(generation_counter, '/', MAX_GENERATION)
		#print_generation(organisms,generation_counter,function)

	# ustawienie wyniku w zależności czego szukaliśmy
	asses(organisms, function)
	if min_max == MAX:
		organisms = sorted(organisms, key=lambda x: x.ocena, reverse=True)
	elif min_max == MIN:
		organisms = sorted(organisms, key=lambda x: x.ocena, reverse=False)
	return organisms[0].data
	
def genetic_func_mean_random4():
	pass	
