import math
import random
#from GradienFunc import gradient


def random_data(domain_list,arg_num):
	data=[None]*arg_num
	for i in range(arg_num):
		data[i]=(random.uniform(domain_list[i][0],domain_list[i][1]))
	return data	
	
def mutate_new_random(generation,domain_list,arg_num):
	mutated=[]
	for org in generation:
		if random.randint(0,100)>95 :
			org.data=random_data(domain_list,arg_num)


def mutate_add_little(generation,domain_list,arg_num):
	mutated=[]
	for org in generation:
		if random.randint(0,100)>95 :
			for i in range(len(org.data)):
				org.data[i]+=0.00001


def mutate_gradient_wise(generation,domain_list,arg_num):
	mutated=[]
	for org in generation:
		if random.randint(0,100)>95 :
			grad=gradient()
			for i in range(len(org.data)):				
				org.data[i]+=grad[i]


#def mutateBits(data,number):
	
	
	
print(math.frexp(16)) #?
	
