from typing import List

import gym

from game.game import Action, AbstractGame
from game.gym_wrappers import DownSampleVisualObservationWrapper


class Centipede(AbstractGame):
    """The Gym Centipede environment"""

    def __init__(self, discount: float):
        super().__init__(discount)
        self.env = gym.make('Centipede-v0')
        self.env = DownSampleVisualObservationWrapper(self.env, factor=5)
        self.actions = list(map(lambda i: Action(i), range(self.env.action_space.n)))
        self.observations = [self.env.reset()]
        self.done = False

    @property
    def action_space_size(self) -> int:
        """Return the size of the action space."""
        return len(self.actions)

    def step(self, action) -> int:
        """Execute one step of the game conditioned by the given action."""

        observation, reward, done, _ = self.env.step(action.index)
        self.observations += [observation]
        self.done = done
        return reward

    def terminal(self) -> bool:
        """Is the game is finished?"""
        return self.done

    def legal_actions(self) -> List[Action]:
        """Return the legal actions available at this instant."""
        return self.actions

    def make_image(self, state_index: int):
        """Compute the state of the game."""
        return self.observations[state_index]
