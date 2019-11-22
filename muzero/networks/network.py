import typing
from typing import Dict, List

from muzero.game.game import Action


class NetworkOutput(typing.NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: List[float]


class Network(object):
    # TODO

    def initial_inference(self, image) -> NetworkOutput:
        # representation + prediction function
        return NetworkOutput(0, 0, {}, [])

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        # dynamics + prediction function
        return NetworkOutput(0, 0, {}, [])

    def get_weights(self):
        # Returns the weights of this networks.
        return []

    def training_steps(self) -> int:
        # How many steps / batches the networks has been trained for.
        return 0
