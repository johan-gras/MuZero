import collections
import math
from typing import Optional, List

import numpy

from muzero.helpers.config import MuZeroConfig
from muzero.pseudocode import softmax_sample
from muzero.game.game import ActionHistory, Player, Action
from muzero.networks.network import NetworkOutput, Network

MAXIMUM_FLOAT_VALUE = float('inf')
KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node(object):

    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


def run_mcts(config: MuZeroConfig, root: Node, action_history: ActionHistory, network: Network):
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """
    min_max_stats = MinMaxStats(config.known_bounds)

    for _ in range(config.num_simulations):
        history = action_history.clone()
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node, min_max_stats)
            history.add_action(action)
            search_path.append(node)

        # Inside the search tree we use the dynamics function to obtain the next
        # hidden state given an action and the previous hidden state.
        parent = search_path[-2]
        network_output = network.recurrent_inference(parent.hidden_state,
                                                     history.last_action())
        expand_node(node, history.to_play(), history.action_space(), network_output)

        backpropagate(search_path, network_output.value, history.to_play(),
                      config.discount, min_max_stats)


def select_action(config: MuZeroConfig, num_moves: int, node: Node, network: Network):
    visit_counts = [
        (child.visit_count, action) for action, child in node.children.items()
    ]
    t = config.visit_softmax_temperature_fn(
        num_moves=num_moves, training_steps=network.training_steps())
    _, action = softmax_sample(visit_counts, t)
    return action


def select_child(config: MuZeroConfig, node: Node, min_max_stats: MinMaxStats):
    """
    Select the child with the highest UCB score.
    """
    _, action, child = max(
        (ucb_score(config, node, child, min_max_stats), action,
         child) for action, child in node.children.items())
    return action, child


def ucb_score(config: MuZeroConfig, parent: Node, child: Node,
              min_max_stats: MinMaxStats) -> float:
    """
    The score for a node is based on its value, plus an exploration bonus based on
    the prior.
    """
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                    config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = min_max_stats.normalize(child.value())
    return prior_score + value_score


def expand_node(node: Node, to_play: Player, actions: List[Action],
                network_output: NetworkOutput):
    """
    We expand a node using the value, reward and policy prediction obtained from
    the neural networks.
    """
    node.to_play = to_play
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward
    policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)


def backpropagate(search_path: List[Node], value: float, to_play: Player,
                  discount: float, min_max_stats: MinMaxStats):
    """
    At the end of a simulation, we propagate the evaluation all the way up the
    tree to the root.
    """
    for node in search_path:
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value


def add_exploration_noise(config: MuZeroConfig, node: Node):
    """
    At the start of each search, we add dirichlet noise to the prior of the root
    to encourage the search to explore new actions.
    """
    actions = list(node.children.keys())
    noise = numpy.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac
