from tests import *

# test_network_simple(net_gradient)
# test_network_simple(net_genetic)


net_data = getIrisDB()
net_model = [4, 5, 5, 2]
nb_tests = 1

plt.figure()
plt.title("Neural network training method comparison")

s, tt = test_network(net_data, net_model, net_gradient, nb_tests=nb_tests)

# steps for gradient
ax = plt.subplot2grid((2, 2), (0, 0))
ax.plot(s)
ax.set(title="Learn Error- gradient", xlabel="iteration", ylabel="average cost")
ax.grid(linestyle='--')
# linia trendu
domain = list(range(len(s)))
z = numpy.polyfit(domain, s, 1)
p = numpy.poly1d(z)
ax.plot(domain, p(domain), "r--")

# time per train for gradient
ax = plt.subplot2grid((2, 2), (0, 1))
ax.plot(tt)
ax.set(title="Time spent on learning- gradient", xlabel="attempt", ylabel="seconds")
ax.grid(linestyle='--')

s, tt = test_network(net_data, net_model, net_genetic, nb_tests=nb_tests)

# steps for genetic
ax = plt.subplot2grid((2, 2), (1, 0))
ax.plot(s)
ax.set(title="Learn Error- genetic", xlabel="generation", ylabel="average cost")
ax.grid(linestyle='--')
# linia trendu
domain = list(range(len(s)))
z = numpy.polyfit(domain, s, 1)
p = numpy.poly1d(z)
ax.plot(domain, p(domain), "r--")

# time per train for gradient
ax = plt.subplot2grid((2, 2), (1, 1))
ax.plot(tt)
ax.set(title="Time spent on learning- genetic", xlabel="attempt", ylabel="seconds")
ax.grid(linestyle='--')

plt.show()
