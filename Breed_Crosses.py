import random
from Organism import organism

#zwraca liste zawierającą średnie po każdej pozycji
def average(list_a, list_b):
	output=[None]*len(list_a)
	for i in range(len(list_a)):
		output[i]=(list_a[i]+list_b[i])/2
	return output
	

def cross_breeds_avg(generacja,probe_num):
	copy=generacja.copy()
	children=[]
	for i in range(int(probe_num/2)):		
		#para bierze losowe 2 organizmy z listy
		parents=random.sample(copy, 2)		
		copy.remove(parents[0])
		copy.remove(parents[1])		
		
		#usrednianie argumentow
		data_a=parents[0].data
		data_b=parents[1].data		
		avg=average(data_a,data_b)		
		
		#tworzenie nowego organizmu
		child=organism(0)		
		child.data=avg	
		children.append(child)						
	return children
