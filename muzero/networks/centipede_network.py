import math

import numpy as np
from .convolutional_networks import *
from tensorflow_core.python.keras import regularizers
from tensorflow_core.python.keras.layers.core import Dense
from tensorflow_core.python.keras.models import Sequential, model_from_json

from game.game import Action
from networks.network import BaseNetwork


class CentipedeNetwork(BaseNetwork):

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 representation_size: int,
                 max_value: int,
                 hidden_neurons: int = 64,
                 weight_decay: float = 1e-4,
                 representation_activation: str = 'tanh',
                 directory: str = None):
        self.state_size = state_size
        self.action_size = action_size
        self.value_support_size = math.ceil(math.sqrt(max_value)) + 1

        if directory is not None:
            print("Loading network from " + directory)
            representation_network = self.load_model(directory + "/representation")
            value_network = self.load_model(directory + "/value")
            policy_network = self.load_model(directory + "/policy")
            dynamic_network = self.load_model(directory + "/dynamic")
            reward_network = self.load_model(directory + "/reward")
        else:
            print("Creating new network")
            regularizer = regularizers.l2(weight_decay)

            # TODO: determine and set input sizes so model can be saved

            representation_network = build_representation_network(50, 32)

            # Shape of representation network's output
            hidden_rep_shape = (3, 2, 6)
            value_network = build_value_network(hidden_rep_shape, regularizer=regularizer)
            policy_network = build_policy_network(hidden_rep_shape, regularizer, self.action_size)

            # Shape when actions are stacked on top of hidden rep
            stacked_hidden_rep_shape = (hidden_rep_shape[0], hidden_rep_shape[1], hidden_rep_shape[2] + 1)
            dynamic_network = build_dynamic_network(stacked_hidden_rep_shape, regularizer=regularizer)
            reward_network = build_reward_network(stacked_hidden_rep_shape, regularizer)

        super().__init__(representation_network, value_network, policy_network, dynamic_network, reward_network)

    def _value_transform(self, value_support: np.array) -> float:
        """
        The value is obtained by first computing the expected value from the discrete support.
        Second, the inverse transform is then apply (the square function).
        """

        value = self._softmax(value_support)
        value = np.dot(value, range(self.value_support_size))
        value = np.asscalar(value) ** 2
        return value

    def _reward_transform(self, reward: np.array) -> float:
        return np.asscalar(reward)

    def _conditioned_hidden_state(self, hidden_state: np.array, action: Action) -> np.array:
        conditioned_hidden = np.concatenate((hidden_state, np.eye(self.action_size)[action.index]))
        return np.expand_dims(conditioned_hidden, axis=0)

    def _softmax(self, values):
        """Compute softmax using numerical stability tricks."""
        values_exp = np.exp(values - np.max(values))
        return values_exp / np.sum(values_exp)
