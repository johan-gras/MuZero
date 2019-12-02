import math
import random

import numpy as np
from tensorflow_core.python.keras import regularizers
from tensorflow_core.python.keras.backend import dot
from tensorflow_core.python.keras.layers.advanced_activations import Softmax
from tensorflow_core.python.keras.layers.core import Dense, Lambda
from tensorflow_core.python.keras.models import Sequential, Model

from game.game import Action
from networks.network import AbstractNetwork, NetworkOutput


class InitialModel(Model):
    def __init__(self, representation_network, value_network, policy_network):
        super(InitialModel, self).__init__()
        self.representation_network = representation_network
        self.value_network = value_network
        self.policy_network = policy_network

    def call(self, image):
        hidden_representation = self.representation_network(image)
        value = self.value_network(hidden_representation)
        policy_logits = self.policy_network(hidden_representation)
        return hidden_representation, value, policy_logits


class DynamicModel(Model):
    def __init__(self, dynamic_network, reward_network, value_network, policy_network):
        super(DynamicModel, self).__init__()
        # self.shared = Sequential([Dense(128, activation='relu')])
        self.dynamic_network = dynamic_network
        self.reward_network = reward_network

        self.value_network = value_network
        self.policy_network = policy_network

    def call(self, conditioned_hidden):
        # x = self.shared(conditioned_hidden)
        hidden_representation = self.dynamic_network(conditioned_hidden)
        reward = self.reward_network(conditioned_hidden)
        value = self.value_network(hidden_representation)
        policy_logits = self.policy_network(hidden_representation)

        return hidden_representation, reward, value, policy_logits


class CartPoleNetwork(AbstractNetwork):
    INPUT_SIZE = 4
    ACTION_SIZE = 2

    # DIMS_REPRESENTATION = [INPUT_SIZE, 128, INPUT_SIZE*2]
    # DIMS_DYNAMIC = [INPUT_SIZE*2 + ACTION_SIZE, 128, INPUT_SIZE*2]
    # DIMS_PREDICTION = [INPUT_SIZE*2, 128, 2 + 1]

    def __init__(self):
        super().__init__()
        regularizer = regularizers.l2(1e-4)
        self.representation_network = Sequential([Dense(64, activation='relu', kernel_regularizer=regularizer),
                                                  Dense(self.INPUT_SIZE, activation='tanh',
                                                        kernel_regularizer=regularizer)])
        self.value_network = Sequential([Dense(64, activation='relu', kernel_regularizer=regularizer),
                                         Dense(24, kernel_regularizer=regularizer)])
        self.policy_network = Sequential([Dense(64, activation='relu', kernel_regularizer=regularizer),
                                          Dense(2, kernel_regularizer=regularizer)])
        self.dynamic_network = Sequential([Dense(64, activation='relu', kernel_regularizer=regularizer),
                                           Dense(self.INPUT_SIZE, activation='tanh', kernel_regularizer=regularizer)])
        self.reward_network = Sequential([Dense(16, activation='relu', kernel_regularizer=regularizer),
                                          Dense(1, kernel_regularizer=regularizer)])

        self.initial_model = InitialModel(self.representation_network, self.value_network, self.policy_network)
        # recurent model?
        self.dynamic_model = DynamicModel(self.dynamic_network, self.reward_network, self.value_network,
                                          self.policy_network)
        self.training_steps = 0

    def softmax(self, values):
        values_exp = np.exp(values - np.max(values))
        return values_exp / np.sum(values_exp)

    def value_transform(self, value):
        value = self.softmax(value)
        value = np.dot(value, range(24))
        value = np.asscalar(value) ** 2
        return value

    def initial_inference(self, image) -> NetworkOutput:
        # representation + prediction function
        hidden_representation, value, policy_logits = self.initial_model.predict(np.expand_dims(image, 0))

        #print(np.min(value), np.max(value), np.min(policy_logits), np.max(policy_logits), np.min(hidden_representation), np.max(hidden_representation))
        value = self.value_transform(value)
        return NetworkOutput(value, 0, self.build_policy_logits(policy_logits), hidden_representation[0])

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        # dynamics + prediction function
        conditioned_hidden = np.concatenate((hidden_state, np.eye(self.ACTION_SIZE)[action.index]))
        conditioned_hidden = np.expand_dims(conditioned_hidden, axis=0)
        hidden_representation, reward, value, policy_logits = self.dynamic_model.predict(conditioned_hidden)

        #print(np.min(value), np.max(value), np.min(policy_logits), np.max(policy_logits), np.min(hidden_representation), np.max(hidden_representation))
        value = self.value_transform(value)
        return NetworkOutput(value, np.asscalar(reward),
                             self.build_policy_logits(policy_logits),
                             hidden_representation[0])

    def build_policy_logits(self, policy_logits):
        return {Action(i): logit for i, logit in enumerate(policy_logits[0])}

    def get_weights(self):
        # Returns the weights of this networks.
        return []

    def cb_get_variables(self):
        def get_variables():
            networks = [self.representation_network, self.value_network, self.policy_network,
                        self.dynamic_network, self.reward_network]
            return [variables
                    for variables_list in map(lambda n: n.weights, networks)
                    for variables in variables_list]

        return get_variables

    # def training_steps(self) -> int:
    #     # How many steps / batches the networks has been trained for.
    #     return 0


class CartPoleNetworkUniform(AbstractNetwork):
    INPUT_SIZE = 4
    ACTION_SIZE = 2

    def __init__(self):
        super().__init__()
        self.training_steps = 0

    def initial_inference(self, image) -> NetworkOutput:
        return NetworkOutput(0, 0, {Action(i): 1 / self.ACTION_SIZE for i in range(self.ACTION_SIZE)},
                             None)

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        # dynamics + prediction function
        return NetworkOutput(0, 0,
                             {Action(i): 1 / self.ACTION_SIZE for i in range(self.ACTION_SIZE)},
                             None)
