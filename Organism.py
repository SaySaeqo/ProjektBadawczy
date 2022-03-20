
#miejsce na różne klasy typu organizm
#może warto będzie stworzyć różne wersje, przechowujące różne info
#typu czas życia. Widziałem jakieś takie rozwiązania

import random
from Constants import *

class organism(object):		
	
	def __init__(self,arg_num):		
		data=[None]*arg_num



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
		
		
		
		
def select_best(parents, children,min_max):
	organisms=parents.copy()+children.copy()
	if min_max == MAX:
		organisms=sorted(organisms, key=lambda x: x.ocena, reverse=True)
	elif min_max == MIN:
		organisms=sorted(organisms, key=lambda x: x.ocena, reverse=False)		
			
	organisms=organisms[:len(organisms)-len(children)]
	return organisms	
