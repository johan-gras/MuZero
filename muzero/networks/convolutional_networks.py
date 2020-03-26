from tensorflow_core.python.keras import Model
from tensorflow_core.python.keras.layers import BatchNormalization, Convolution2D, Input
from tensorflow_core.python.keras.layers import Conv2D, AveragePooling2D
from tensorflow_core.python.keras.layers.core import Activation, Layer
from tensorflow_core.python.keras.layers import Add


'''
Residual unit modified and updated to account for being keras 2.0 from https://github.com/relh/keras-residual-unit
'''

def conv_block(feat_maps_out, prev):
    prev = BatchNormalization(axis=-1)(prev) # Specifying the axis and mode allows for later merging
    prev = Activation('relu')(prev)
    prev = Conv2D(filters=feat_maps_out, kernel_size=3, padding='same')(prev)
    prev = BatchNormalization(axis=-1)(prev) # Specifying the axis and mode allows for later merging
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


def build_dynamics_network():
    pass


def build_representation_network(img_row, img_col):
    # TODO: does RGB come before or after?
    shape = (img_row, img_col, 3)
    input = Input(shape)
    c1 = Conv2D(filters=3, kernel_size=3, strides=2, padding='same', activation='relu', input_shape=shape)(input)
    r1 = residual(3, 3, c1)
    r2 = residual(3, 3, r1)
    c2 = Conv2D(filters=6, kernel_size=3, strides=2, padding='same', activation='relu', input_shape=(img_row/2, img_col/2, 3))(r2)
    r3 = residual(6, 6, c2)
    r4 = residual(6, 6, r3)
    r5 = residual(6, 6, r4)
    a1 = AveragePooling2D(strides=2)(r5)
    r6 = residual(6, 6, a1)
    r7 = residual(6, 6, r6)
    r8 = residual(6, 6, r7)
    a2 = AveragePooling2D(strides=2)(r8)
    model = Model(inputs=input, outputs=a2)
    # model.summary()
    # plot_model(model, './conv.png')
    return model
