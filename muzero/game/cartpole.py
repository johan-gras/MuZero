from typing import List

from game.game import Action, AbstractGame
import gym

from game.gym_wrappers import NormalizedObservationWrapper


class CartPole(AbstractGame):
    """The Gym CartPole environment"""

    def __init__(self, discount: float):
        super().__init__(discount)
        self.env = gym.make('CartPole-v1')
        self.env = NormalizedObservationWrapper(self.env, low=[-2.4, -2.0, -0.42, -3.5], high=[2.4, 2.0, 0.42, 3.5])
        self.actions = list(map(lambda i: Action(i), range(self.env.action_space.n)))
        self.observations = [self.env.reset()]
        self.done = False

    @property
    def action_space_size(self) -> int:
        return len(self.actions)

    def step(self, action) -> int:
        # self.env.render()
        observation, reward, done, _ = self.env.step(action.index)
        self.observations += [observation]
        self.done = done
        # if done:
        #     self.env.close()
        return reward

    def terminal(self) -> bool:
        return self.done

    def legal_actions(self) -> List[Action]:
        return self.actions

    def make_image(self, state_index: int):
        return self.observations[state_index]
