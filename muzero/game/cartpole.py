from typing import List

from muzero.game.game import Action, Game
import gym

from muzero.game.gym_wrappers import NormalizedObservationWrapper


class CartPole(Game):
    """The Gym CartPole environment"""

    def __init__(self, discount: float):
        super().__init__(discount)
        self.env = gym.make('CartPole-v0')
        self.env = NormalizedObservationWrapper(self.env, low=[-2.4, -2.0, -0.42, -3.5], high=[2.4, 2.0, 0.42, 3.5])
        self.actions = list(range(self.env.action_space.n))
        self.observations = [self.env.reset()]
        self.done = False

    @property
    def action_space_size(self) -> int:
        return len(self.actions)

    def step(self, action) -> int:
        observation, reward, done = self.env.step(action.index)
        self.observations += observation
        self.done = done
        return reward

    def terminal(self) -> bool:
        return self.done

    def legal_actions(self) -> List[Action]:
        return self.actions

    def make_image(self, state_index: int):
        # TODO: should I include the action?
        return self.observations[state_index]
