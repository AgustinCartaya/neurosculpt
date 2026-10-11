import tensorflow as tf
import unet3d as unet3d
from tensorflow.keras.layers import Activation, Dense, GlobalAveragePooling3D


tf.keras.backend.set_image_data_format("channels_first")

# tf.keras.backend.set_floatx('float16')
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

def get_end_net(basic_net, nb_classes):
    x_feat = basic_net.output
    x_gap = GlobalAveragePooling3D()(x_feat)
    x = Dense(nb_classes, name='dense_logits')(x_gap)
    x = Activation('softmax', dtype='float32', name='classification_output')(x)
    return x

def create_model(input_shape, nb_classes, weight_dir=None):
    # weight_dir = f"{pil.get_base_folder()}genesis_model/chest_weights/Genesis_Chest_CT.h5"
    models_genesis = unet3d.unet_model_3d(input_shape, batch_normalization=True)

    if weight_dir is not None:
        models_genesis.load_weights(weight_dir)

    encoder_output_layer = models_genesis.get_layer('depth_7_relu')
    y_pred = get_end_net(encoder_output_layer, nb_classes)

    encoder_model = tf.keras.Model(inputs=models_genesis.input, outputs=y_pred)

    return encoder_model


# weight_dir = "C:/Users/Agustin/Master/Master/S4/code/test_code/NeuroSculpt/clustering/naive_clustering/genesis_model/Genesis_Chest_CT.h5"
# model = create_model((1, 128, 128, 128), nb_classes=16, weight_dir=weight_dir)
# print(model.summary())

# Total params: 7,030,656
# Trainable params: 7,027,776
# Non-trainable params: 2,880

