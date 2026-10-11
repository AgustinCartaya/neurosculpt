"""Models Genesis encoder features used by the MPGAN perceptual loss."""
import tensorflow as tf

from brainpred.methods.gan.mpgan import unet3d


def create_features_extractor(input_shape, weight_dir=None):
    """Return a model mapping a channels-first volume to 4 intermediate Models Genesis activations.

    Args:
        input_shape: ``(1, x, y, z)``.
        weight_dir: path to ``Genesis_Chest_CT.h5`` (see ``paths.get_genesis_weights_path_name()``);
            ``None`` keeps random weights.
    """
    models_genesis = unet3d.unet_model_3d(input_shape, batch_normalization=True)
    if weight_dir is not None:
        models_genesis.load_weights(weight_dir)

    # NOTE: the second feature map is 'depth_2_relu' although the original variable was named depth_3_relu.
    layer_names = ['depth_1_relu', 'depth_2_relu', 'depth_5_relu', 'depth_7_relu']
    outputs = [models_genesis.get_layer(name).output for name in layer_names]
    return tf.keras.Model(inputs=models_genesis.input, outputs=outputs)
