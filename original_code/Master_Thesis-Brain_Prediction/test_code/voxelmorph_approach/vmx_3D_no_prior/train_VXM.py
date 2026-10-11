import os
import sys

current_path = os.path.abspath(os.getcwd())
base_path = os.path.abspath(os.path.join(current_path, '../../../'))
sys.path.append(base_path)

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

import numpy as np
from tqdm import tqdm
import datetime

from data_loaders.data_loader_complete import DataLoaderComplete
import project_images_loader as pil
import matplotlib.pyplot as plt

import utils.functions as fc
import utils.visualization as vis
import utils.util_deep_learning as udl
import utils.util as util

import voxelmorph as vxm


def show_generated_image(id, vxm_model, img_size, current_plane=116):
    def fast_resieze(img, img_size):
        return img[:img_size[0], :img_size[1], :img_size[2]]
    
    img_hunt3 = fast_resieze(pil.load_preprocessed_image(3, id, absolute=True)[0], img_size)
    img_hunt4 = fast_resieze(pil.load_preprocessed_image(4, id, absolute=True)[0], img_size)

    img_gen_h4 = vxm_model(np.expand_dims(np.expand_dims(img_hunt3, axis=0), axis=-1), training=False)
    img_gen_h4 = img_gen_h4[0][0, ...,0].numpy()
    img_gen_h4[img_gen_h4 < 0] = 0
    img_gen_h4 = img_gen_h4 / np.max(img_gen_h4)
    return vis.shows_3_images(img_hunt3, img_hunt4, img_gen_h4)


def create_vxm_model(img_shape):
    # configure unet input shape (concatenation of moving and fixed images)
    ndim = len(img_shape)
    unet_input_features = 1
    inshape = (*img_shape, unet_input_features)

    # configure unet features 
    nb_features = [
        [32, 64, 128, 256],         # encoder features
        [256, 128, 64, 32, 32, 16]  # decoder features
    ]
    # build model
    unet = vxm.networks.Unet(inshape=inshape, nb_features=nb_features)

    if ndim == 2:
        disp_tensor = tf.keras.layers.Conv2D(ndim, kernel_size=3, padding='same', name='disp')(unet.output)
    elif ndim == 3:
        disp_tensor = tf.keras.layers.Conv3D(ndim, kernel_size=3, padding='same', name='disp')(unet.output)
 

    # warp the moving image with the transformer
    spatial_transformer = vxm.layers.SpatialTransformer(name='transformer')
    moving_image = tf.expand_dims(unet.input[..., 0], axis=-1)
    moved_image_tensor = spatial_transformer([moving_image, disp_tensor])

    outputs = [moved_image_tensor, disp_tensor]
    vxm_model = tf.keras.models.Model(inputs=unet.inputs, outputs=outputs)
    return vxm_model


mse_loss = vxm.losses.MSE().loss
dfield_loss = vxm.losses.Grad('l2').loss

def losses(h4_batch, h4_moved, dfield):
    # mse loss
    loss_mse = mse_loss(h4_batch, h4_moved)

    # gradient loss
    loss_dfield = tf.reduce_mean(dfield_loss(None, dfield))
    loss = loss_mse + 0.05 * loss_dfield
    return loss, loss_mse, loss_dfield

@tf.function
def train_step(vxm_model, h3_batch, h4_batch, optimizer):
    with tf.GradientTape() as model_tape:
        h4_moved, dfield = vxm_model(h3_batch, training=True)
        loss, loss_mse, loss_dfield = losses(h4_batch, h4_moved, dfield)

    gradients = model_tape.gradient(loss, vxm_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vxm_model.trainable_variables))

    return loss, loss_mse, loss_dfield



def train(num_epochs, batch_size):
    # input_shape = (193,229,193)
    input_shape = (192,224,192)
    # input_shape = (64,64,64)
    current_plane = 116

    # ---- Models
    vxm_model = create_vxm_model(input_shape)

    # optimizers
    inital_lr = 1e-4
    optimizer = tf.keras.optimizers.Adam(inital_lr)

    # ---- Data
    server = True
    data_train = DataLoaderComplete(image_sufix=pil.INAME_PREPROCESSED,
                                    batch_size=batch_size, 
                                    image_size=input_shape, 
                                    channels_first=False,
                                    shuffle=True, 
                                    nb_patients=300, 
                                    shuffle_before_selecting=False, 
                                    server=server)
    # visualization
    nb_loss = 3
    acum_loss_train = udl.AcumulativeMetrics(nb_metrics = nb_loss)

    log_dir=f"{current_path}/models/logs/train"
    summary_writer = tf.summary.create_file_writer(log_dir)

    # ---- checkpoint
    checkpoint_dir = f"{current_path}/models/training_checkpoints/"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=optimizer,
                                    vxm_model=vxm_model)
    # restore checkpoint
    # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    for epoch in range(0,num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # visualize images
        val_results = show_generated_image("08202", vxm_model, input_shape, current_plane=current_plane)

        # ---- Train
        acum_loss_train.reset()
        progress_bar = tqdm(total=len(data_train), desc=f'Initializing...')
        for step, (h3_batch, h4_batch) in enumerate(data_train):
            loss, loss_mse, loss_dfield = train_step(vxm_model, h3_batch, h4_batch, optimizer)
            acum_loss_train.update((loss, loss_mse, loss_dfield))
            losses_text = acum_loss_train.get_progress_bar_text(['loss', 'loss_mse', 'loss_dfield'])

            progress_bar.set_description(desc=losses_text)
            progress_bar.update(1)
        del progress_bar

        # ---- Check points
        if  epoch > 0 and epoch % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        
        # ---- Learning rate
        if epoch > 10 and epoch % 4 == 0:
            inital_lr *= 0.5
            optimizer.learning_rate = inital_lr

        # ---- tensorboard
        loss_mean = acum_loss_train.get_mean()
        with summary_writer.as_default():
            tf.summary.scalar('loss', loss_mean[0], step=epoch)
            tf.summary.scalar('loss_mse', loss_mean[0], step=epoch)
            tf.summary.scalar('loss_dfield', loss_mean[0], step=epoch)

            # images
            gen_image = np.expand_dims(np.expand_dims(val_results[2], axis=-1), axis=0)
            hunt4_vs_gen = np.expand_dims(np.stack([val_results[2], val_results[1], np.zeros_like(val_results[1])], axis=-1), axis=0)
            tf.summary.image("generated", gen_image, step=epoch)
            tf.summary.image("R: Gen - G: HUNT4", hunt4_vs_gen, step=epoch)
    

train(num_epochs=50, batch_size=1)