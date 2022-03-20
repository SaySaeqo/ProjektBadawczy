import Constants
import numpy as np
from Genetic_Algorithm import randomData


def gradient(args):  # rosenbrock gradient (hardcoded)
    x1 = -400 * args[0] * args[1] + 400 * (args[0] ** 3) + 2 * args[0] - 2
    x2 = 200 * args[1] - 200 * (args[0] ** 2)
    return [x1, x2]


# argument 'function' na razie nie używany bo brakuje funkcji, któraby dla dowolnej funkcji
# wyliczała gradient
def gradient_func(function, arg_num, domain_list, min_max, learn_rate, max_iter=1000, tol=0.1):
    start = randomData(domain_list, arg_num)
    x = np.array(start)
    tol = [tol] * arg_num
    tol = np.array(tol)

    iters = 0
    for _ in range(max_iter):
        print(x)
        iters += 1
        diff = gradient(x.tolist())
        diff = list(map(lambda a: learn_rate * a, diff))
        if all((np.abs(diff) < tol).tolist()):
            break
        if min_max == Constants.MIN:
            x = x - np.array(diff)
        else:  # MAX
            x = x + np.array(diff)

    return iters, x.tolist()


print(gradient_func(0, 2, [[-2, 2], [-2, 2]], Constants.MIN, 0.0005, 10000, 0.0001))
