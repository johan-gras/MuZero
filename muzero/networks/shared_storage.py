from networks.cartpole_network import CartPoleNetworkUniform, CartPoleNetwork
from networks.network import AbstractNetwork


class SharedStorage(object):

    def __init__(self):
        self._networks = {}
        self.current_network = None

    def latest_network(self) -> AbstractNetwork:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            # return make_uniform_network()
            return CartPoleNetworkUniform()

    def save_network(self, step: int, network: AbstractNetwork):
        self._networks[step] = network
