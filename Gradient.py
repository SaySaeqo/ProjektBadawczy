



def derivative(func,args,n):
    h=0.000001
    new_args=args.copy()
    new_args[n]+=h
    return (func(new_args)-func(args))/h




def gradient(func,args_values):
    result=[];
    for i in range(len(args_values)):
        result.append(derivative(func,args_values,i))
    return result
