from Tests import  *

#przyklad wywołania genetycznego



func=get_function(4)
#nie działa, bo get function nie zwraca dziedziny!
#print("wynik",genetic_func_mean_change(func,2,domain,MIN,PROBE_NUMBER))

#print("wynik",genetic_func_mean_gradient_change_litle(func.function, func.arg_num, func.domain, func.min_max, PROBE_NUMBER))
#print(gradient_algorithm(func.function,func.arg_num,func.domain,func.min_max,100))

test_algorithms()

#problematyczna funkcja to ta o numerze 2 (Rosenbrocka!)
#najrzadziej daje poprawne wyniki!