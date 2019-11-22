from muzero.helpers.config import MuZeroConfig
from muzero.pseudocode import launch_job
from muzero.networks.shared_storage import SharedStorage
from muzero.self_play.self_play import run_selfplay
from muzero.training.training import train_network
from muzero.training.replay_buffer import ReplayBuffer


def muzero(config: MuZeroConfig):
    """
    MuZero training is split into two independent parts: Network training and
    self-play data generation.
    These two parts only communicate by transferring the latest networks checkpoint
    from the training to the self-play, and the finished games from the self-play
    to the training.
    """
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)

    for _ in range(config.num_actors):
        launch_job(run_selfplay, config, storage, replay_buffer)

    train_network(config, storage, replay_buffer)

    return storage.latest_network()