import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import tensorflow_addons as tfa

keras = tf.keras
layers = keras.layers
RANDOM_SEED = 36

tf.keras.backend.set_image_data_format("channels_first")
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')


def create_generator_model(input_shape):
    kernel_initializer = keras.initializers.glorot_uniform(seed=RANDOM_SEED) 

    # final layer
    inputs = keras.Input(shape=(1, *input_shape))

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
    output = layers.Activation('sigmoid', dtype='float32')(x)
    
    model = keras.Model(inputs=inputs, outputs=output)
    return model


def generate_encoder_base_block(filters, kernel_initializer, x, max_pooling=True):
    x = layers.Conv3D(filters[0], 3, padding='same', strides=(1, 1, 1), kernel_initializer=kernel_initializer)(x)
    x = tfa.layers.InstanceNormalization(axis=1)(x)
    # x = layers.GroupNormalization(groups=-1, axis=1)(x)
    x = layers.ReLU()(x)

    x = layers.Conv3D(filters[1], 3, padding='same', strides=(1, 1, 1), kernel_initializer=kernel_initializer)(x)
    x = tfa.layers.InstanceNormalization(axis=1)(x)
    # x = layers.GroupNormalization(groups=-1, axis=1)(x)
    x = layers.ReLU()(x)
    x_concat = x
    if max_pooling:
        x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    return x, x_concat


def generate_decoder_base_block(filters, kernel_initializer, x, x_concat):
    x = layers.UpSampling3D(size=(2, 2, 2))(x)
    x = layers.concatenate([x, x_concat], axis=1)

    x = layers.Conv3D(filters, 3, padding='same', strides=(1, 1, 1), kernel_initializer=kernel_initializer)(x)
    x = tfa.layers.InstanceNormalization(axis=1)(x)
    # x = layers.GroupNormalization(groups=-1, axis=1)(x)
    x = layers.ReLU()(x)

    x = layers.Conv3D(filters, 3, padding='same', strides=(1, 1, 1), kernel_initializer=kernel_initializer)(x)
    x = tfa.layers.InstanceNormalization(axis=1)(x)
    # x = layers.GroupNormalization(groups=-1, axis=1)(x)
    x = layers.ReLU()(x)
    return x


def visualize_model():
    import numpy as np
    patch_size= (64,64,64)
    model = create_generator_model(patch_size)
    print(model.summary())
    img_sample = np.random.rand(1, 1, *patch_size).astype(np.float32)
    res = model.predict(img_sample)

# visualize_model()