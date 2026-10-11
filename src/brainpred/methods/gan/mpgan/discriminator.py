"""MPGAN discriminator: PatchGAN-style 3D CNN on channels-first volumes."""
import tensorflow as tf

from brainpred.methods.gan.layers.normalization import NORM_TFA, instance_norm

keras = tf.keras
layers = keras.layers


def create_discriminator_model(input_shape, norm=NORM_TFA):
    """
    Args:
        norm: ``"tfa"`` (MPGAN_paper_16) or ``"group"`` (MPGAN_paper_16_seg_loss).
            NOTE: the original tfa variant normalized with ``axis=-1`` (the last *spatial* axis, since
            data is channels-first) while the group variant used ``axis=1`` (channels). Both are kept
            as trained, so the two variants are not numerically equivalent.
    """
    norm_axis = -1 if norm == NORM_TFA else 1
    constant_initializer = tf.keras.initializers.Constant(0)
    kernel_initializer = tf.compat.v1.keras.initializers.RandomNormal(mean=0, stddev=0.02)

    inputs = tf.keras.Input(shape=(1, *input_shape))

    x = generate_base_block(64, kernel_initializer, inputs, norm, norm_axis)
    x = generate_base_block(128, kernel_initializer, x, norm, norm_axis)
    x = generate_base_block(256, kernel_initializer, x, norm, norm_axis)
    x = generate_base_block(512, kernel_initializer, x, norm, norm_axis)

    x = layers.Conv3D(1, 4, strides=(1, 1, 1), padding='same', kernel_initializer=kernel_initializer,
                      bias_initializer=constant_initializer)(x)
    outputs = layers.Activation('sigmoid', dtype='float32')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def generate_base_block(filters, kernel_initializer, x, norm=NORM_TFA, norm_axis=-1):
    x = layers.Conv3D(filters, 4, strides=(2, 2, 2), padding='same', kernel_initializer=kernel_initializer)(x)
    x = instance_norm(norm, axis=norm_axis)(x)
    return layers.LeakyReLU(0.2)(x)
