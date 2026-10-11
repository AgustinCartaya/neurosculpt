
import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../../'))

sys.path.append(base_path)

import project_images_loader as pil

import tensorflow as tf
import unet3d as unet3d

tf.keras.backend.set_image_data_format("channels_first")
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

def create_features_extractor(input_shape, weight_dir=None):
    # weight_dir = f"{pil.get_base_folder()}genesis_model/chest_weights/Genesis_Chest_CT.h5"
    models_genesis = unet3d.unet_model_3d(input_shape, batch_normalization=True)

    if weight_dir is not None:
        models_genesis.load_weights(weight_dir)


    depth_1_relu = models_genesis.get_layer('depth_1_relu').output
    depth_3_relu = models_genesis.get_layer('depth_2_relu').output
    depth_5_relu = models_genesis.get_layer('depth_5_relu').output
    depth_7_relu = models_genesis.get_layer('depth_7_relu').output

    outputs = [depth_1_relu, depth_3_relu, depth_5_relu, depth_7_relu]
    encoder_model = tf.keras.Model(inputs=models_genesis.input, outputs=outputs)

    return encoder_model



# model = create_features_extractor((1, 64, 64, 64), pil.get_genesis_weights_path_name())
# print(model.summary())



