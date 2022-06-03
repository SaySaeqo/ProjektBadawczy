from tests import *

# test_network_simple(net_gradient)
# test_network_simple(net_genetic)

test_iris()
plt.show()
test_raisin()
plt.show()
# test_beans()
# plt.show()

# db = get_iris_db()
# s, tt = test_network(db,[4,10,10,10,10,2],train_func=net_genetic_wlt, nb_tests=1, test_data_length=10)
# plt.plot(s)
# plt.savefig("last.png")