from tests import *

# test_network_simple(net_gradient)
# test_network_simple(net_genetic)


net_data = getIrisDB()
net_model = [4, 5, 5, 2]
nb_tests = 10

plot_network_comparison(net_data,net_model,nb_tests)
plt.savefig("last.png")
plt.show()