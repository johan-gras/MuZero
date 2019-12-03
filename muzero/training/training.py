"""Training module: this is where MuZero neurons are trained."""

import math
from itertools import zip_longest

import numpy as np
import tensorflow_core as tf

from helpers.config import MuZeroConfig
from networks.cartpole_network import CartPoleNetwork
from networks.network import BaseNetwork
from networks.shared_storage import SharedStorage
from training.replay_buffer import ReplayBuffer


def train_network(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer, epochs: int):
    # if not storage.current_network:
    #     # learning_rate = config.lr_init * config.lr_decay_rate ** (tf.train.get_global_step() / config.lr_decay_steps)
    #     storage.current_network = CartPoleNetwork()
    #     # storage.current_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.025) # 0.05
    #     storage.current_optimizer = tf.keras.optimizers.SGD(learning_rate=0.05, momentum=config.momentum)
    network = storage.current_network
    optimizer = storage.optimizer

    for _ in range(epochs):
        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        update_weights_batch(optimizer, network, batch, config.weight_decay)
        storage.save_network(network.training_steps, network)


def update_weights_batch(optimizer: tf.keras.optimizers, network: BaseNetwork, batch, weight_decay: float):
    # TODO: Scale properly the gradient in the batch

    def scale_gradient(tensor, scale: float):
        """Trick function to scale the gradient in tensorflow"""
        return (1. - scale) * tf.stop_gradient(tensor) + scale * tensor

    def loss():
        # Pre-processing batch
        loss = 0
        gradient_scale = 1. / 5  # On average 5 actions during BPTT
        image_batch, actions_time_batch, targets_batch = zip(*batch)
        actions_time_batch = list(zip_longest(*actions_time_batch, fillvalue=None))

        # Building batch of valid actions
        # And a dynamic mask for hidden representations during BPTT
        mask_time_batch = []
        dynamic_mask_time_batch = []
        last_mask = [True]*len(image_batch)
        for i, actions_batch in enumerate(actions_time_batch):
            mask = list(map(lambda a: bool(a), actions_batch))
            dynamic_mask = [now for last, now in zip(last_mask, mask) if last]
            mask_time_batch.append(mask)
            dynamic_mask_time_batch.append(dynamic_mask)
            last_mask = mask
            actions_time_batch[i] = [action.index for action in actions_batch if action]

        # Initial step, from the real observation.
        targets_init_batch, *targets_time_batch = zip(*targets_batch)
        representation_batch, value_batch, policy_batch = network.initial_model(np.array(image_batch))

        # Compute the loss of the first pass
        target_value_batch, _, target_policy_batch = zip(*targets_init_batch)
        mask_policy = list(map(lambda l: bool(l), target_policy_batch))
        target_policy_batch = list(filter(lambda l: bool(l), target_policy_batch))
        policy_batch = tf.boolean_mask(policy_batch, mask_policy)

        loss += tf.math.reduce_mean(loss_value_batch(target_value_batch, value_batch))
        loss += tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=policy_batch, labels=target_policy_batch))

        # Recurrent steps, from action and previous hidden state.
        for actions_batch, targets_batch, mask, dynamic_mask in zip(actions_time_batch, targets_time_batch, mask_time_batch, dynamic_mask_time_batch):
            target_value_batch, target_reward_batch, target_policy_batch = zip(*targets_batch)
            # Only execute BPTT for elements with an action or with a policy target
            representation_batch = tf.boolean_mask(representation_batch, dynamic_mask)
            target_value_batch = tf.boolean_mask(target_value_batch, mask)
            target_reward_batch = tf.boolean_mask(target_reward_batch, mask)
            # Shit happens
            target_policy_batch = [policy for policy, b in zip(target_policy_batch, mask) if b]
            mask_policy = list(map(lambda l: bool(l), target_policy_batch))
            target_policy_batch = tf.convert_to_tensor([policy for policy in target_policy_batch if policy])

            # Concatenate representations with actions batch
            actions_batch = tf.one_hot(actions_batch, network.action_size)
            conditioned_representation_batch = tf.concat((representation_batch, actions_batch), axis=1)
            representation_batch, reward_batch, value_batch, policy_batch = network.recurrent_model(conditioned_representation_batch)
            policy_batch = tf.boolean_mask(policy_batch, mask_policy)

            # Compute the partial loss
            # TODO: deal with the scale gradient stuff
            l = (tf.math.reduce_mean(loss_value_batch(target_value_batch, value_batch)) +
                 tf.math.reduce_mean(tf.math.squared_difference(target_reward_batch, tf.squeeze(reward_batch))) +
                 tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=policy_batch, labels=target_policy_batch)))
            loss += scale_gradient(l, gradient_scale)

            # Half the gradient
            representation_batch = scale_gradient(representation_batch, 0.5)

        return loss

    optimizer.minimize(loss=loss, var_list=network.cb_get_variables())
    network.training_steps += 1


def loss_value_batch(target_value_batch, value_batch):
    # TODO: make this less cartpole specific.

    batch_size = len(target_value_batch)
    targets = np.zeros((batch_size, 24))
    sqrt_value = np.sqrt(target_value_batch)
    floor_value = np.floor(sqrt_value).astype(int)
    rest = sqrt_value - floor_value
    targets[range(batch_size), floor_value.astype(int)] = 1 - rest
    targets[range(batch_size), floor_value.astype(int) + 1] = rest

    return tf.nn.softmax_cross_entropy_with_logits(logits=value_batch, labels=targets)


# Depracated

def update_weights(optimizer: tf.keras.optimizers, network: BaseNetwork, batch, weight_decay: float):
    def scale_gradient(tensor, scale: float):
        """Trick function to scale the gradient in tensorflow"""
        return (1. - scale) * tf.stop_gradient(tensor) + scale * tensor

    def loss():
        loss = 0
        for image, actions, targets in batch:
            # Initial step, from the real observation.
            hidden_representation, value, policy_logits = network.initial_model(np.expand_dims(image, 0))
            reward = tf.constant([[0.]])
            gradient_scale = 1.0
            predictions = [(gradient_scale, value, reward, policy_logits)]

            # Recurrent steps, from action and previous hidden state.
            for action in actions:
                encoded_action = tf.expand_dims(tf.one_hot(action.index, network.ACTION_SIZE), axis=0)
                conditioned_hidden = tf.concat((hidden_representation, encoded_action), axis=1)
                hidden_representation, reward, value, policy_logits = network.recurrent_model(conditioned_hidden)
                predictions.append((1.0 / len(actions), value, reward, policy_logits))

                # Half the gradient
                hidden_representation = scale_gradient(hidden_representation, 0.5)

            for prediction, target in zip(predictions, targets):
                gradient_scale, value, reward, policy_logits = prediction
                target_value, target_reward, target_policy = target
                l = (
                        loss_value(target_value, value) +
                        tf.keras.losses.MSE(reward, tf.convert_to_tensor(target_reward)) +
                        tf.nn.softmax_cross_entropy_with_logits(logits=policy_logits,
                                                                labels=tf.convert_to_tensor([target_policy])) if target_policy else 0.
                )
                loss += scale_gradient(l, gradient_scale)

        return loss

    optimizer.minimize(loss=loss, var_list=network.cb_get_variables())
    network.training_steps += 1


def loss_value(target_value, logits_value):
    target = np.zeros(24)
    value = math.sqrt(target_value)
    floor = math.floor(value)
    rest = value - floor
    target[floor] = 1 - rest
    target[floor+1] = rest

    return tf.nn.softmax_cross_entropy_with_logits(logits=logits_value, labels=tf.convert_to_tensor([target]))
