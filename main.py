from tests import *

# test_network_simple(net_gradient)
# test_network_simple(net_genetic)

test_iris(10)
plt.show()
test_raisin(10)
plt.show()
test_beans(10)
plt.show()

# db = get_iris_db()
# history, time_passed = test_network(db,[4,10,10,10,10,2], net_genetic_wlt, test_data_length=10)
# plt.plot(history["av_costs"])
# plt.savefig("last.png")