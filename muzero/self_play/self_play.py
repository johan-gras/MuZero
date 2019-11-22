"""Self-Play module"""

from muzero.helpers.config import MuZeroConfig
from muzero.training.replay_buffer import ReplayBuffer
from muzero.game.game import Game
from muzero.networks.network import Network
from muzero.networks.shared_storage import SharedStorage
from muzero.self_play.mcts import run_mcts, select_action, expand_node, add_exploration_noise, Node


def run_selfplay(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer):
    """
    Each self-play job is independent of all others; it takes the latest networks
    snapshot, produces a game and makes it available to the training job by
    writing it to a shared replay buffer.
    """
    while True:
        network = storage.latest_network()
        game = play_game(config, network)
        replay_buffer.save_game(game)


def play_game(config: MuZeroConfig, network: Network) -> Game:
    """
    Each game is produced by starting at the initial board position, then
    repeatedly executing a Monte Carlo Tree Search to generate moves until the end
    of the game is reached.
    """
    game = config.new_game()

    while not game.terminal() and len(game.history) < config.max_moves:
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        current_observation = game.make_image(-1)
        expand_node(root, game.to_play(), game.legal_actions(),
                    network.initial_inference(current_observation))
        add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the networks.
        run_mcts(config, root, game.action_history(), network)
        action = select_action(config, len(game.history), root, network)
        game.apply(action)
        game.store_search_statistics(root)
    return game
