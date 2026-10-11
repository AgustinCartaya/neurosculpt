"""3D U-Net of Models Genesis (Zhou et al.), used only as frozen feature extractor for the perceptual loss."""
import tensorflow as tf

keras = tf.keras
layers = keras.layers
RANDOM_SEED = 36


def unet_model_3d(input_shape, pool_size=(2, 2, 2), n_labels=1, deconvolution=False,
                  depth=4, n_base_filters=32, batch_normalization=False, activation_name="sigmoid"):
    """Build the 3D U-Net (channels-first).

    Args:
        input_shape: ``(n_channels, x, y, z)``; spatial sizes must be divisible by ``pool_size ** depth``.
        pool_size: pool size of the max-pooling operations.
        n_labels: number of output channels.
        deconvolution: use transposed convolutions instead of up-sampling.
        depth: number of resolution levels.
        n_base_filters: filters of the first layer (doubled at every level).
        batch_normalization: add batch normalization after every convolution.
        activation_name: activation of the output layer.
    """
    inputs = keras.Input(input_shape)
    current_layer = inputs
    levels = []
    num_layer = 0

    # Contracting path
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters * (2 ** layer_depth),
                                          batch_normalization=batch_normalization, layer_depth=num_layer)
        num_layer += 1
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters * (2 ** layer_depth) * 2,
                                          batch_normalization=batch_normalization, layer_depth=num_layer)
        num_layer += 1
        if layer_depth < depth - 1:
            current_layer = layers.MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # Expanding path
    for layer_depth in range(depth - 2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer.shape[1])(current_layer)
        concat = layers.concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1].shape[1], layer_depth=num_layer,
                                                 input_layer=concat, batch_normalization=batch_normalization)
        num_layer += 1
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1].shape[1], layer_depth=num_layer,
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)
        num_layer += 1

    final_convolution = layers.Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = layers.Activation(activation_name)(final_convolution)
    return keras.Model(inputs=inputs, outputs=act)


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), layer_depth=None):
    """Conv3D (+ BatchNorm) + activation; layers are named ``depth_<i>_{conv,bn,relu}``."""
    layer = layers.Conv3D(n_filters, kernel, padding=padding, strides=strides,
                          name=f"depth_{layer_depth}_conv")(input_layer)
    if batch_normalization:
        layer = layers.BatchNormalization(axis=1, name=f"depth_{layer_depth}_bn")(layer)
    if activation is None:
        return layers.Activation('relu', name=f"depth_{layer_depth}_relu")(layer)
    return activation()(layer)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False, trainable=True):
    if deconvolution:
        return layers.Conv3DTranspose(filters=n_filters, kernel_size=kernel_size,
                                      strides=strides, trainable=trainable,
                                      kernel_initializer=keras.initializers.glorot_uniform(seed=RANDOM_SEED))
    return layers.UpSampling3D(size=pool_size)
