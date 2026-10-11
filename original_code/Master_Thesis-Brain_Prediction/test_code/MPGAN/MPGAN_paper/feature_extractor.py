import tensorflow as tf
import unet3d as unet3d
tf.keras.backend.set_image_data_format("channels_first")

def create_features_extractor(input_shape, weight_dir):
    # weight_dir = f"{pil.get_base_folder()}genesis_model/chest_weights/Genesis_Chest_CT.h5"
    models_genesis = unet3d.unet_model_3d(input_shape, batch_normalization=True)
    models_genesis.load_weights(weight_dir)

    encoder_output_layer = models_genesis.get_layer('depth_7_relu')
    encoder_model = tf.keras.Model(inputs=models_genesis.input, outputs=encoder_output_layer.output)

    return encoder_model





