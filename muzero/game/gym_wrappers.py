import gym
import numpy as np
import cv2


class ScalingObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper that apply a min-max scaling of observations.
    """

    def __init__(self, env, low=None, high=None):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)

        low = np.array(self.observation_space.low if low is None else low)
        high = np.array(self.observation_space.high if high is None else high)

        self.mean = (high + low) / 2
        self.max = high - self.mean

    def observation(self, observation):
        return (observation - self.mean) / self.max


class DownSampleVisualObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper that significantly down-samples the gym image for more efficient network training
    """

    def __init__(self, env, factor=0):
        super().__init__(env)
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        self.factor = factor

    def observation(self, observation):
        # TODO: Implement down-sampling effectively
        # Default size is (250, 160, 3)
        if self.factor > 0:
            return cv2.resize(observation, dsize=(int(250/self.factor), int(80/self.factor)), interpolation=cv2.INTER_CUBIC)
        else:
            return observation
