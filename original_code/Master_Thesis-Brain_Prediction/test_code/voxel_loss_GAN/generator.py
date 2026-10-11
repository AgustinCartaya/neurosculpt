import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import tensorflow_addons as tfa

keras = tf.keras
layers = keras.layers
RANDOM_SEED = 36


def create_generator_model(input_shape):
    kernel_initializer = keras.initializers.glorot_uniform(seed=RANDOM_SEED) 

    # final layer
    inputs = keras.Input(shape=(*input_shape, 1))

    # Encoder part
    x, x_concat1 = generate_encoder_base_block(filters=(20,40), kernel_initializer=kernel_initializer, x=inputs)
    x, x_concat2 = generate_encoder_base_block(filters=(40,80), kernel_initializer=kernel_initializer, x=x)
    x, x_concat3 = generate_encoder_base_block(filters=(80,160), kernel_initializer=kernel_initializer, x=x)
    x, _ = generate_encoder_base_block(filters=(160,320), kernel_initializer=kernel_initializer, x=x, max_pooling=False)

    # Decoder part
    x = generate_decoder_base_block(filters=160, kernel_initializer=kernel_initializer, x=x, x_concat=x_concat3)
    x = generate_decoder_base_block(filters=80, kernel_initializer=kernel_initializer, x=x, x_concat=x_concat2)
    x = generate_decoder_base_block(filters=40, kernel_initializer=kernel_initializer, x=x, x_concat=x_concat1)

    # final layer
    x = layers.Conv3D(1, (1, 1, 1), kernel_initializer=kernel_initializer)(x)
    output = layers.Activation("tanh")(x)
    
    model = keras.Model(inputs=inputs, outputs=output)
    return model


def generate_encoder_base_block(filters, kernel_initializer, x, max_pooling=True):
    x = layers.Conv3D(filters[0], 3, padding='same', strides=(1, 1, 1), kernel_initializer=kernel_initializer)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = layers.ReLU()(x)

    x = layers.Conv3D(filters[1], 3, padding='same', strides=(1, 1, 1), kernel_initializer=kernel_initializer)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = layers.ReLU()(x)
    x_concat = x
    if max_pooling:
        x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    return x, x_concat


def generate_decoder_base_block(filters, kernel_initializer, x, x_concat):
    x = layers.UpSampling3D(size=(2, 2, 2))(x)
    x = layers.concatenate([x, x_concat], axis=-1)

    x = layers.Conv3D(filters, 3, padding='same', strides=(1, 1, 1), kernel_initializer=kernel_initializer)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = layers.ReLU()(x)

    x = layers.Conv3D(filters, 3, padding='same', strides=(1, 1, 1), kernel_initializer=kernel_initializer)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = layers.ReLU()(x)
    return x

