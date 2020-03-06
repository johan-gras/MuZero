from config import MuZeroConfig, make_cartpole_config
from networks.shared_storage import SharedStorage
from self_play.self_play import run_selfplay, run_eval
from training.replay_buffer import ReplayBuffer
from training.training import train_network
import sys


def muzero(config: MuZeroConfig, save_directory: str, load_directory: str):
    """
    MuZero training is split into two independent parts: Network training and
    self-play data generation.
    These two parts only communicate by transferring the latest networks checkpoint
    from the training to the self-play, and the finished games from the self-play
    to the training.
    In contrast to the original MuZero algorithm this version doesn't works with
    multiple threads, therefore the training and self-play is done alternately.
    """

    if load_directory is not None:
        # User specified directory to load network from
        network = config.old_network(load_directory)
    else:
        network = config.new_network()

    # TODO: figure out whether new uniform network and new optimizer are OK for loading previous model
    storage = SharedStorage(network, config.uniform_network(), config.new_optimizer(), save_directory)
    replay_buffer = ReplayBuffer(config)

    for loop in range(config.nb_training_loop):
        print("Training loop", loop)
        score_train = run_selfplay(config, storage, replay_buffer, config.nb_episodes)
        train_network(config, storage, replay_buffer, config.nb_epochs)

        print("Train score:", score_train)
        print("Eval score:", run_eval(config, storage, 50))
        print(f"MuZero played {config.nb_episodes * (loop + 1)} "
              f"episodes and trained for {config.nb_epochs * (loop + 1)} epochs.\n")

    return storage.latest_network()


if __name__ == '__main__':
    # Train the model with given config
    config = make_cartpole_config()

    save_directory = None
    load_directory = None
    if len(sys.argv) > 1:
        save_directory = sys.argv[1]
        load_directory = sys.argv[2]
    muzero(config, save_directory, load_directory)
