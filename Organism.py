
#miejsce na różne klasy typu organizm
#może warto będzie stworzyć różne wersje, przechowujące różne info
#typu czas życia. Widziałem jakieś takie rozwiązania

import random
from Constants import *

class organism(object):		
	
	def __init__(self,arg_num):		
		data=[None]*arg_num



def create_population(arg_num,domain_list,probe_num):
	org=[]
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



#TO DO ----------------------------------------------------------

#niestety to będzie działać tylko dla liczb wartości nieujemnych.
#póki co w ogóle nie działa XD
def select_roulette(parents,children,min_max):
	'''
	organizmy są już ocenione
	:param parents: organizmy
	:param children: organizmy
	:param min_max: tryb
	:return: new_organisms size of parents
	'''
	organisms = parents.copy() + children.copy()
	#stwórz koło

	#liczymy sume ocen
	sum_asses=0
	for i in range(len(organisms)):
		sum_asses+=organisms[i].ocena

	#ustawiamy szacowaną szanse na bycie wylosowanym
	chances=[] # i-ta szansa odpowiada itemu organizmowu w organisms
	if min_max == MIN:
		pass
	elif min_max == MAX:
		for i in range(len(organisms)):
			chances.append( organisms[i].ocena/sum_asses)

	#n razy wylosuj organizm
	for i in range(len(parents)):
		pass


	organisms = organisms[:len(organisms) - len(children)]
	return organisms


def select_tournament(parents,children,min_max):
	organisms = parents.copy() + children.copy()
	GROUP_SIZE=3
	new_generation=[]
	# wybierz n/2 razy 3 organizmy
	for i in range(int(len(parents)/2)):
		fighters = random.sample(organisms, GROUP_SIZE)
		#usuń wybrańców z oryginału
		for j in range(GROUP_SIZE):
			organisms.remove(fighters[j])

		#posortuj
		if min_max == MAX:
			fighters = sorted(fighters, key=lambda x: x.ocena, reverse=True)
		elif min_max == MIN:
			fighters = sorted(fighters, key=lambda x: x.ocena, reverse=False)
		#usun najgorszego
		fighters=fighters[:len(fighters)-1] #1 jest odrzucany
		#dodaj 2  najbardziej pożadane
		new_generation=new_generation+fighters



	return new_generation