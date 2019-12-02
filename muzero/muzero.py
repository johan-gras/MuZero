from helpers.config import MuZeroConfig, make_atari_config, make_cartpole_config
from networks.cartpole_network import CartPoleNetwork
from pseudocode import launch_job
from networks.shared_storage import SharedStorage
from self_play.self_play import run_selfplay, run_eval
from training.training import train_network
from training.replay_buffer import ReplayBuffer


# TODO
# - Plot result
# - Scale properly the gradient in the batch


def muzero(config: MuZeroConfig):
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)

    train_episodes = 50
    eval_episodes = 100
    epochs = 50

    for loop in range(100):
        print("Training loop", loop)
        score_eval = run_eval(config, storage, eval_episodes)
        score_train = run_selfplay(config, storage, replay_buffer, train_episodes)
        train_network(config, storage, replay_buffer, epochs)

        print("Eval score:", score_eval)
        print("Train score:", score_train)
        print(f"Played {train_episodes * (loop + 1)} episodes and trained for {epochs * (loop + 1)} epochs.\n")

    return storage.latest_network()


def muzero_multiple_workers(config: MuZeroConfig):
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


if __name__ == '__main__':
    config = make_cartpole_config()
    muzero(config)
