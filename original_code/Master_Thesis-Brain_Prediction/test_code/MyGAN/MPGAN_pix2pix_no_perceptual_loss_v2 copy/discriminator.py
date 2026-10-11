import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.layers import Dropout


keras = tf.keras
layers = keras.layers

dropout_rate = 0.0

def create_discriminator_model(input_shape):
    constant_initializer = tf.keras.initializers.Constant(0)
    kernel_initializer = tf.compat.v1.keras.initializers.RandomNormal(mean=0, stddev=0.02)
    
    inp = tf.keras.Input(shape=(*input_shape, 1), name='input_image')
    tar = tf.keras.Input(shape=(*input_shape, 1), name='target_image')

    inputs = tf.keras.layers.concatenate([inp, tar])
    # inputs = inp
    
    x = generate_base_block(filters=64, kernel_initializer=kernel_initializer, x=inputs)
    x = Dropout(dropout_rate)(x)  # Añade dropout después del primer bloque
    x = generate_base_block(filters=128, kernel_initializer=kernel_initializer, x=x)
    x = Dropout(dropout_rate)(x)  # Añade dropout después del segundo bloque
    x = generate_base_block(filters=256, kernel_initializer=kernel_initializer, x=x)
    x = Dropout(dropout_rate)(x)  # Añade dropout después del tercer bloque
    x = generate_base_block(filters=512, kernel_initializer=kernel_initializer, x=x)
    x = Dropout(dropout_rate)(x) 

    output = layers.Conv3D(1, 4, strides=(1, 1, 1), activation='sigmoid', padding='same', kernel_initializer=kernel_initializer,
                      bias_initializer=constant_initializer)(x)

    return tf.keras.Model(inputs=[inp, tar], outputs=output)


def generate_base_block(filters, kernel_initializer, x):
    x = layers.Conv3D(filters, 4, strides=(2, 2, 2), padding='same', kernel_initializer=kernel_initializer)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = tf.nn.leaky_relu(x, 0.2)
    return x




