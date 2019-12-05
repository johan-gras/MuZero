import random
from itertools import zip_longest
from typing import List

from config import MuZeroConfig
from game.game import AbstractGame


class ReplayBuffer(object):

    def __init__(self, config: MuZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        # Generate some sample of data to train on
        games = self.sample_games()
        game_pos = [(g, self.sample_position(g)) for g in games]
        game_data = [(g.make_image(i), g.history[i:i + num_unroll_steps],
                      g.make_target(i, num_unroll_steps, td_steps, g.to_play()))
                     for (g, i) in game_pos]

        # Pre-process the batch
        image_batch, actions_time_batch, targets_batch = zip(*game_data)
        targets_init_batch, *targets_time_batch = zip(*targets_batch)
        actions_time_batch = list(zip_longest(*actions_time_batch, fillvalue=None))

        # Building batch of valid actions and a dynamic mask for hidden representations during BPTT
        mask_time_batch = []
        dynamic_mask_time_batch = []
        last_mask = [True] * len(image_batch)
        for i, actions_batch in enumerate(actions_time_batch):
            mask = list(map(lambda a: bool(a), actions_batch))
            dynamic_mask = [now for last, now in zip(last_mask, mask) if last]
            mask_time_batch.append(mask)
            dynamic_mask_time_batch.append(dynamic_mask)
            last_mask = mask
            actions_time_batch[i] = [action.index for action in actions_batch if action]

        batch = image_batch, targets_init_batch, targets_time_batch, actions_time_batch, mask_time_batch, dynamic_mask_time_batch
        return batch

    def sample_games(self) -> List[AbstractGame]:
        # Sample game from buffer either uniformly or according to some priority.
        return random.choices(self.buffer, k=self.batch_size)

    def sample_position(self, game: AbstractGame) -> int:
        # Sample position from game either uniformly or according to some priority.
        return random.randint(0, len(game.history))
