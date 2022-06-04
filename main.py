from tests import *

# test_network_simple(net_gradient)
# test_network_simple(net_genetic)

test_iris()
plt.show()
test_raisin()
plt.show()
test_beans()
plt.show()

# db = get_iris_db()
# history, time_passed = test_network(db,[4,10,10,10,10,2], net_genetic_wlt, test_data_length=10)
# plt.plot(history["av_costs"])
# plt.savefig("last.png")