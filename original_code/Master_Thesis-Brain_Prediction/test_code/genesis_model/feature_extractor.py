import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import tensorflow_addons as tfa

keras = tf.keras
layers = keras.layers
RANDOM_SEED = 36


def create_features_extractor(input_shape):
    kernel_initializer = keras.initializers.glorot_uniform(seed=RANDOM_SEED) 

    inputs = keras.Input(shape=(*input_shape, 1))

    x = generate_encoder_base_block(filters=(34,64), kernel_initializer=kernel_initializer, x=inputs)
    x = generate_encoder_base_block(filters=(64,128), kernel_initializer=kernel_initializer, x=x)
    x = generate_encoder_base_block(filters=(128,256), kernel_initializer=kernel_initializer, x=x)
    output = generate_encoder_base_block(filters=(256,512), kernel_initializer=kernel_initializer, x=x, max_pooling=False)

    model = keras.Model(inputs=inputs, outputs=output)

    return model


def generate_encoder_base_block(filters, kernel_initializer, x, max_pooling=True):
    x = layers.Conv3D(filters[0], 3, padding='same', strides=(1, 1, 1), kernel_initializer=kernel_initializer)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = layers.ReLU()(x)

    x = layers.Conv3D(filters[1], 3, padding='same', strides=(1, 1, 1), kernel_initializer=kernel_initializer)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = layers.ReLU()(x)
    if max_pooling:
        x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    return x



