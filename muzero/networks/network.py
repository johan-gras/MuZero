import typing
from abc import ABC, abstractmethod
from typing import Dict, List, Callable

import numpy as np
from tensorflow.keras.models import Model

from game.game import Action


class NetworkOutput(typing.NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: typing.Optional[List[float]]

    @staticmethod
    def build_policy_logits(policy_logits):
        return {Action(i): logit for i, logit in enumerate(policy_logits[0])}


class AbstractNetwork(ABC):

    def __init__(self):
        self.training_steps = 0

    @abstractmethod
    def initial_inference(self, image) -> NetworkOutput:
        pass

    @abstractmethod
    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        pass


class UniformNetwork(AbstractNetwork):
    """policy -> uniform, value -> 0, reward -> 0"""

    def __init__(self, action_size: int):
        super().__init__()
        self.action_size = action_size

    def initial_inference(self, image) -> NetworkOutput:
        return NetworkOutput(0, 0, {Action(i): 1 / self.action_size for i in range(self.action_size)}, None)

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        return NetworkOutput(0, 0, {Action(i): 1 / self.action_size for i in range(self.action_size)}, None)


class InitialModel(Model):
    """Model that combine the representation and prediction (value+policy) network."""

    def __init__(self, representation_network: Model, value_network: Model, policy_network: Model):
        super(InitialModel, self).__init__()
        self.representation_network = representation_network
        self.value_network = value_network
        self.policy_network = policy_network

    def call(self, image):
        hidden_representation = self.representation_network(image)
        value = self.value_network(hidden_representation)
        policy_logits = self.policy_network(hidden_representation)
        return hidden_representation, value, policy_logits


class RecurrentModel(Model):
    """Model that combine the dynamic, reward and prediction (value+policy) network."""

    def __init__(self, dynamic_network: Model, reward_network: Model, value_network: Model, policy_network: Model):
        super(RecurrentModel, self).__init__()
        self.dynamic_network = dynamic_network
        self.reward_network = reward_network
        self.value_network = value_network
        self.policy_network = policy_network

    def call(self, conditioned_hidden):
        hidden_representation = self.dynamic_network(conditioned_hidden)
        reward = self.reward_network(conditioned_hidden)
        value = self.value_network(hidden_representation)
        policy_logits = self.policy_network(hidden_representation)
        return hidden_representation, reward, value, policy_logits


class BaseNetwork(AbstractNetwork):
    """Base class that contains all the networks and models of MuZero."""

    def __init__(self, representation_network: Model, value_network: Model, policy_network: Model,
                 dynamic_network: Model, reward_network: Model):
        super().__init__()
        # Networks blocks
        self.representation_network = representation_network
        self.value_network = value_network
        self.policy_network = policy_network
        self.dynamic_network = dynamic_network
        self.reward_network = reward_network

        # Models for inference and training
        self.initial_model = InitialModel(self.representation_network, self.value_network, self.policy_network)
        self.recurrent_model = RecurrentModel(self.dynamic_network, self.reward_network, self.value_network,
                                              self.policy_network)

    def initial_inference(self, image: np.array) -> NetworkOutput:
        """representation + prediction function"""

        hidden_representation, value, policy_logits = self.initial_model.predict(np.expand_dims(image, 0))
        output = NetworkOutput(value=self._value_transform(value),
                               reward=0.,
                               policy_logits=NetworkOutput.build_policy_logits(policy_logits),
                               hidden_state=hidden_representation[0])
        return output

    def recurrent_inference(self, hidden_state: np.array, action: Action) -> NetworkOutput:
        """dynamics + prediction function"""

        conditioned_hidden = self._conditioned_hidden_state(hidden_state, action)
        hidden_representation, reward, value, policy_logits = self.recurrent_model.predict(conditioned_hidden)
        output = NetworkOutput(value=self._value_transform(value),
                               reward=self._reward_transform(reward),
                               policy_logits=NetworkOutput.build_policy_logits(policy_logits),
                               hidden_state=hidden_representation[0])
        return output

    @abstractmethod
    def _value_transform(self, value: np.array) -> float:
        pass

    @abstractmethod
    def _reward_transform(self, reward: np.array) -> float:
        pass

    @abstractmethod
    def _conditioned_hidden_state(self, hidden_state: np.array, action: Action) -> np.array:
        pass

    def cb_get_variables(self) -> Callable:
        """Return a callback that return the trainable variables of the network."""

        def get_variables():
            networks = (self.representation_network, self.value_network, self.policy_network,
                        self.dynamic_network, self.reward_network)
            return [variables
                    for variables_list in map(lambda n: n.weights, networks)
                    for variables in variables_list]

        return get_variables
