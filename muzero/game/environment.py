from abc import ABC, abstractmethod
from typing import List


class Action(object):

    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index


class Environment(ABC):
    """The environment MuZero is interacting with."""

    def step(self, action):
        pass

    def terminal(self) -> bool:
        # Game specific termination rules.
        pass

    def legal_actions(self) -> List[Action]:
        # Game specific calculation of legal actions.
        pass

    def make_image(self, state_index: int):
        # Game specific feature planes.
        pass
