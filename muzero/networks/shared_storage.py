import tensorflow as tf

from networks.network import BaseNetwork, UniformNetwork, AbstractNetwork


class SharedStorage(object):
    """Save the different versions of the network."""

    def __init__(self, network: BaseNetwork, uniform_network: UniformNetwork, optimizer: tf.keras.optimizers):
        self._networks = {}
        self.current_network = network
        self.uniform_network = uniform_network
        self.optimizer = optimizer

    def latest_network(self) -> AbstractNetwork:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return self.uniform_network

    def save_network(self, step: int, network: BaseNetwork):
        self._networks[step] = network
