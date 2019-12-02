import typing
from abc import ABC, abstractmethod
from typing import Dict, List

from game.game import Action


class NetworkOutput(typing.NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: List[float]


class AbstractNetwork(ABC):

    def __init__(self):
        self.representation_network = None
        self.dynamic_model = None
        self.prediction_network = None

    @abstractmethod
    def initial_inference(self, image) -> NetworkOutput:
        # representation + prediction function
        return NetworkOutput(0, 0, {}, [])

    @abstractmethod
    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        # dynamics + prediction function
        return NetworkOutput(0, 0, {}, [])

    # @abstractmethod
    # def get_weights(self):
    #     # Returns the weights of this networks.
    #     return []

    # @abstractmethod
    # def training_steps(self) -> int:
    #     # How many steps / batches the networks has been trained for.
    #     return 0
