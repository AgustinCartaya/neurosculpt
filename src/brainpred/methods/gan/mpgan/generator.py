"""MPGAN generator: 3D U-Net on whole channels-first volumes ``(batch, 1, 192, 224, 192)``.

Requires ``tf.keras.backend.set_image_data_format("channels_first")`` before building
(see :func:`brainpred.methods.gan.variants.configure_tensorflow`).
"""
import tensorflow as tf

from brainpred.methods.gan.layers.normalization import NORM_TFA, instance_norm

keras = tf.keras
layers = keras.layers
RANDOM_SEED = 36


def create_generator_model(input_shape, norm=NORM_TFA):
    kernel_initializer = keras.initializers.glorot_uniform(seed=RANDOM_SEED)

    inputs = keras.Input(shape=(1, *input_shape))

    # Encoder
    x, x_concat1 = generate_encoder_base_block((20, 40), kernel_initializer, inputs, norm)
    x, x_concat2 = generate_encoder_base_block((40, 80), kernel_initializer, x, norm)
    x, x_concat3 = generate_encoder_base_block((80, 160), kernel_initializer, x, norm)
    x, _ = generate_encoder_base_block((160, 320), kernel_initializer, x, norm, max_pooling=False)

    # Decoder
    x = generate_decoder_base_block(160, kernel_initializer, x, x_concat3, norm)
    x = generate_decoder_base_block(80, kernel_initializer, x, x_concat2, norm)
    x = generate_decoder_base_block(40, kernel_initializer, x, x_concat1, norm)

    # Output
    x = layers.Conv3D(1, (1, 1, 1), kernel_initializer=kernel_initializer)(x)
    output = layers.Activation('sigmoid', dtype='float32')(x)
    return keras.Model(inputs=inputs, outputs=output)


def _conv_norm_relu(filters, kernel_initializer, x, norm):
    x = layers.Conv3D(filters, 3, padding='same', strides=(1, 1, 1), kernel_initializer=kernel_initializer)(x)
    x = instance_norm(norm, axis=1)(x)
    return layers.ReLU()(x)


def generate_encoder_base_block(filters, kernel_initializer, x, norm=NORM_TFA, max_pooling=True):
    x = _conv_norm_relu(filters[0], kernel_initializer, x, norm)
    x = _conv_norm_relu(filters[1], kernel_initializer, x, norm)
    x_concat = x
    if max_pooling:
        x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    return x, x_concat


def generate_decoder_base_block(filters, kernel_initializer, x, x_concat, norm=NORM_TFA):
    x = layers.UpSampling3D(size=(2, 2, 2))(x)
    x = layers.concatenate([x, x_concat], axis=1)
    x = _conv_norm_relu(filters, kernel_initializer, x, norm)
    x = _conv_norm_relu(filters, kernel_initializer, x, norm)
    return x
