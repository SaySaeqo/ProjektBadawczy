from Functions import get_function
from Genetic_Algorithms import *
from Constants import *
from Gradient_Algorithm import *

#przyklad wywołania genetycznego
#print("wynik",genetic_func_mean_change(get_function(2),2,[[-10,10],[-10,10]],MIN,PROBE_NUMBER))

#print("wynik",genetic_func_mean_gradient_change_litle(get_function(1),2,[[-10,10],[-10,10],[-10,10]],MIN,PROBE_NUMBER))

print(gradient_algorithm(get_function(2),2,[[-10,10],[-10,10]],MIN,100))



#problematyczna funkcja to ta o numerze 2 (Rosenbrocka!)
#najrzadziej daje poprawne wyniki!