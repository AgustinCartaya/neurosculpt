import tensorflow as tf
# from tensorflow.keras.layers import Conv3D, LeakyReLU, BatchNormalization, Flatten
# from tensorflow_addons.layers import InstanceNormalization

import voxelmorph as vxm


def create_vxm_model(img_shape):
    # configure unet input shape (concatenation of moving and fixed images)
    inshape = (*img_shape, 2)

    # configure unet features 
    # nb_features = [[32, 64, 64, 64, 64, 64, 64],
    #                 [64, 64, 64, 64, 64, 64, 32]]
    
    nb_features = [[32, 64, 64, 64, 64],
                [64, 64, 64, 64, 32]]
    

    # build model
    unet = vxm.networks.Unet(inshape=inshape, nb_features=nb_features)
    disp_tensor = tf.keras.layers.Conv3D(3, kernel_size=3, padding='same', name='disp')(unet.output)

    # warp the moving image with the transformer
    spatial_transformer = vxm.layers.SpatialTransformer(name='transformer')
    moving_image = tf.expand_dims(unet.input[..., 0], axis=-1)
    moved_image_tensor = spatial_transformer([moving_image, disp_tensor])

    outputs = [moved_image_tensor, disp_tensor]
    vxm_model = tf.keras.models.Model(inputs=unet.inputs, outputs=outputs)
    return vxm_model




# model = create_vxm_model((192,224, 192))
# res = model(tf.random.normal((1, 192, 224, 192, 2)))
# print(model.summary())
