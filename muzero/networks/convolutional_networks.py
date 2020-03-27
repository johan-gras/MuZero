from tensorflow_core.python.keras import Model
from tensorflow_core.python.keras.layers import AveragePooling2D, Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU
from tensorflow_core.python.keras.layers.core import Activation


'''
Residual unit modified and updated to account for being keras 2.0 from https://github.com/relh/keras-residual-unit
'''


def conv_block(feat_maps_out, prev):
    prev = BatchNormalization(axis=-1)(prev)  # Specifying the axis and mode allows for later merging
    prev = Activation('relu')(prev)
    prev = Conv2D(filters=feat_maps_out, kernel_size=3, padding='same')(prev)
    prev = BatchNormalization(axis=-1)(prev)  # Specifying the axis and mode allows for later merging
    prev = Activation('relu')(prev)
    prev = Conv2D(filters=feat_maps_out, kernel_size=3, padding='same')(prev)
    return prev


def skip_block(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = Conv2D(filters=feat_maps_out, kernel_size=1, padding='same')(prev)
    return prev


def residual(feat_maps_in, feat_maps_out, prev_layer):
    '''
    A customizable residual unit with convolutional and shortcut blocks
    Args:
      feat_maps_in: number of channels/filters coming in, from input or previous layer
      feat_maps_out: how many output channels/filters this block will produce
      prev_layer: the previous layer
    '''

    skip = skip_block(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block(feat_maps_out, prev_layer)

    merged = Add()([skip, conv])
    return merged


def build_dynamic_network(shape, regularizer):
    return None


def build_reward_network(shape, regularizer):
    """
    network = Sequential([
        Dense(16, activation='relu', kernel_regularizer=regularizer),
        Dense(1, kernel_regularizer=regularizer)
    ])
    return reward_network
    """
    return None


def build_policy_network(shape, regularizer, action_size):
    policy_input = Input(shape)
    c1 = Conv2D(filters=1, kernel_size=1, padding='same', activation='linear')(policy_input)
    b1 = BatchNormalization(axis=-1)(c1)
    l1 = LeakyReLU()(b1)
    f1 = Flatten()(l1)
    d1 = Dense(action_size, use_bias=False, activation='linear')(f1)
    policy_model = Model(inputs=policy_input, outputs=d1)
    return policy_model


def build_value_network(shape, regularizer):
    value_input = Input(shape)
    c1 = Conv2D(filters=1, kernel_size=1, padding='same', activation='linear')(value_input)
    b1 = BatchNormalization(axis=-1)(c1)
    l1 = LeakyReLU()(b1)
    f1 = Flatten()(l1)
    d2 = Dense(20, use_bias=False, activation='linear')(f1)
    l2 = LeakyReLU()(d2)
    d2 = Dense(1, use_bias=False, activation='tanh')(l2)
    value_model = Model(inputs=value_input, outputs=d2)
    return value_model


def build_representation_network(img_row, img_col, filter_size1=3, filter_size2=6, conv_strides=1, avg_pool_strides=2):
    # TODO: does RGB come before or after?
    shape = (img_row, img_col, 3)
    input = Input(shape)
    c1 = Conv2D(filters=filter_size1, kernel_size=3, strides=conv_strides, padding='same', activation='relu',
                input_shape=shape)(input)

    r1 = residual(filter_size1, filter_size1, c1)
    r2 = residual(filter_size1, filter_size1, r1)

    c2 = Conv2D(filters=filter_size2, kernel_size=3, strides=conv_strides, padding='same', activation='relu',
                input_shape=(img_row/conv_strides, img_col/conv_strides, 3))(r2)

    r3 = residual(filter_size2, filter_size2, c2)
    r4 = residual(filter_size2, filter_size2, r3)
    r5 = residual(filter_size2, filter_size2, r4)

    a1 = AveragePooling2D(strides=avg_pool_strides)(r5)

    r6 = residual(filter_size2, filter_size2, a1)
    r7 = residual(filter_size2, filter_size2, r6)
    r8 = residual(filter_size2, filter_size2, r7)

    a2 = AveragePooling2D(strides=avg_pool_strides)(r8)

    model = Model(inputs=input, outputs=a2)
    return model
