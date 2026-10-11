import tensorflow as tf
import tensorflow_addons as tfa

keras = tf.keras
layers = keras.layers

tf.keras.backend.set_image_data_format("channels_first")


def create_discriminator_model(input_shape):
    constant_initializer = tf.keras.initializers.Constant(0)
    kernel_initializer = tf.compat.v1.keras.initializers.RandomNormal(mean=0, stddev=0.02)

    inputs = tf.keras.Input(shape=(1, *input_shape))

    x = generate_base_block(filters=64, kernel_initializer=kernel_initializer, x=inputs)
    x = generate_base_block(filters=128, kernel_initializer=kernel_initializer, x=x)
    x = generate_base_block(filters=256, kernel_initializer=kernel_initializer, x=x)
    x = generate_base_block(filters=512, kernel_initializer=kernel_initializer, x=x)

    x = layers.Conv3D(1, 4, strides=(1, 1, 1), padding='same', kernel_initializer=kernel_initializer,
                      bias_initializer=constant_initializer)(x)

    outputs = tf.sigmoid(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def generate_base_block(filters, kernel_initializer, x):
    x = layers.Conv3D(filters, 4, strides=(2, 2, 2), padding='same', kernel_initializer=kernel_initializer)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    # x = layers.GroupNormalization(groups=-1, axis=1)(x)
    # x = tf.nn.leaky_relu(x, 0.2)
    x = layers.LeakyReLU(0.2)(x)
    return x



def visualize_model():
    import numpy as np

    patch_size= (64,64,64)
    model = create_discriminator_model(patch_size)
    print(model.summary())
    img_sample = np.random.rand(1, 1, *patch_size).astype(np.float32)
    res = model.predict(img_sample)

# visualize_model()
