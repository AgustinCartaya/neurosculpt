import os
import sys

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../../'))
sys.path.append(base_path)


import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

import numpy as np
from tqdm import tqdm
import datetime

from data_loaders.data_loader_patches import DataLoaderPatches
import project_images_loader as pil
import matplotlib.pyplot as plt

import discriminator as Discriminator
import generator as Generator
from layers.Wavelet3D_16 import DWT3D

import utils.functions as fc
import utils.util_deep_learning as udl
import utils.visualization as vis
import utils.util as util
import utils.patches_controller as patches_controller
import utils.util_freesurfer_segmentation as ufs

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')





def get_weight_mask_hunt4(data_train, weights, dtype=np.float32):
    bmask_hunt3, bmask_hunt4 = data_train.get_iter_custom_patches("bmask")
    seg_hunt3, seg_hunt4 = data_train.get_iter_custom_patches("seg3")
    
    weigth_mask_h3 = np.zeros(bmask_hunt3.shape, dtype=dtype)
    weigth_mask_h4 = np.zeros(bmask_hunt4.shape, dtype=dtype)

    for i in range(len(weigth_mask_h3)):
        weigth_mask_h3[i,...,0] = ufs.get_4_tissues_prob_mask(bmask_hunt3[i,...,0], seg_hunt3[i,...,0], weights)
        weigth_mask_h4[i,...,0] = ufs.get_4_tissues_prob_mask(bmask_hunt4[i,...,0], seg_hunt4[i,...,0], weights)
        
    return weigth_mask_h3, weigth_mask_h4









def generate_image_step(img_org, G, patch_size, patch_step, batch_size=20):
    img_patches = patches_controller.extract_patches(img_org, patch_shape=patch_size, extraction_step=patch_step).astype(np.float32)
    gen_patches = np.zeros_like(img_patches)

    for i in tqdm(range(0, len(img_patches), batch_size)):
        patches_batch = np.expand_dims(img_patches[i:min(i+batch_size, len(img_patches))], axis=-1)
        gen_result = G(patches_batch, training=False)
        gen_patches[i:min(i+batch_size, len(img_patches))] = gen_result[-1][..., 0]

    gen_image = patches_controller.perform_voting(gen_patches, 
                                                  patch_shape=patch_size, 
                                                  expected_shape=img_org.shape, 
                                                  extraction_step=patch_step,
                                                  window_sep=patch_step)
    return gen_image


def show_generated_image(id, G_h4, G_h3, patch_size, patch_step, current_plane=116, show=True):
    img_hunt3, _ = pil.load_preprocessed_image(3, id, absolute=True)
    img_hunt4, _ = pil.load_preprocessed_image(4, id, absolute=True)

    img_gen_h4 = generate_image_step(img_hunt3, G_h4, patch_size=patch_size, patch_step=patch_step)
    img_gen_h3 = generate_image_step(img_hunt4, G_h3, patch_size=patch_size, patch_step=patch_step)

    _, _, img_gen_h4 = vis.shows_3_images(img_hunt3, img_hunt4, img_gen_h4, current_plane=current_plane, show=show)
    img_hunt3, img_hunt4, img_gen_h3 = vis.shows_3_images(img_hunt3, img_hunt4, img_gen_h3, current_plane=current_plane, show=show)

    return img_hunt3, img_hunt4, img_gen_h4, img_gen_h3









cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# def adversarial_loss(disc_h4_real, disc_h3_real, disc_h4_fake, disc_h3_fake):
#     loss = (
#         tf.reduce_mean(tf.math.log(disc_h3_real)) +
#         tf.reduce_mean(tf.math.log(1.0 - disc_h3_fake)) +
#         tf.reduce_mean(tf.math.log(disc_h4_real)) +
#         tf.reduce_mean(tf.math.log(1.0 - disc_h4_fake))
#     )
#     return loss

def adversarial_loss(disc_h4_real, disc_h3_real, disc_h4_fake, disc_h3_fake):

    real_loss_h3 = cross_entropy(tf.ones_like(disc_h3_real), disc_h3_real)
    real_loss_h4 = cross_entropy(tf.ones_like(disc_h4_real), disc_h4_real)

    fake_loss_h3 = cross_entropy(tf.zeros_like(disc_h3_fake), disc_h3_fake)
    fake_loss_h4 = cross_entropy(tf.zeros_like(disc_h4_fake), disc_h4_fake)

    loss = real_loss_h3 + real_loss_h4 + fake_loss_h3 + fake_loss_h4
    return loss

def quality_loss(target_image, predicted_image, disc_prob, beta=1.5):
    loss = tf.reduce_mean(tf.abs(target_image - predicted_image) * tf.pow(1 - disc_prob, beta))
    return loss

def gram_matrix(input_tensor):
    batch_size, width, height, deep, channels = input_tensor.get_shape().as_list()
    reshaped_tensor = tf.reshape(input_tensor, [batch_size, width * height, deep])
    gram_matrix = tf.matmul(reshaped_tensor, reshaped_tensor, transpose_a=True)
    num_elements = tf.cast(width * height * deep * channels, tf.float32)
    gram_matrix /= num_elements
    return gram_matrix


def texture_loss(target_image, predicted_image):
    gram_real = gram_matrix(target_image)
    gram_fake = gram_matrix(predicted_image)
    
    loss = tf.reduce_mean(tf.square(gram_real - gram_fake))
    
    return loss

dwt3D_layer = DWT3D(wavelet_name="bior1.3")
def frequency_loss(target_image, predicted_image):
    dwt_target_image = dwt3D_layer(target_image)
    dwt_predicted_image = dwt3D_layer(predicted_image)
    loss = tf.reduce_mean(tf.abs(dwt_target_image - dwt_predicted_image))
    return tf.cast(loss, tf.float32)

def calc_cycle_loss(target_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(target_image - cycled_image))
    return loss1

















@tf.function
def train_step(G_h3, D_h3, G_h4, D_h4, h3_batch, h4_batch, g_h3_optimizer, d_h3_optimizer, g_h4_optimizer, d_h4_optimizer, h3_batch_bmask, h4_batch_bmask):
    with tf.GradientTape() as g_h4_tape, tf.GradientTape() as d_h4_tape, tf.GradientTape() as g_h3_tape, tf.GradientTape() as d_h3_tape:
        h4_fake1, h4_fake2, h4_fake3  = G_h4(h3_batch, training=True)
        h3_fake1_cyc, h3_fake2_cyc, h3_fake3_cyc = G_h3(h4_fake3, training=True)

        h3_fake1, h3_fake2, h3_fake3  = G_h3(h4_batch, training=True)
        h4_fake1_cyc, h4_fake2_cyc, h4_fake3_cyc = G_h4(h3_fake3, training=True)
        

        disc_h4_real1, disc_h4_real2, disc_h4_real3 = D_h4(h4_batch, training=True)
        disc_h4_fake1, disc_h4_fake2, disc_h4_fake3 = D_h4(h4_fake3, training=True)

        disc_h3_real1, disc_h3_real2, disc_h3_real3 = D_h3(h3_batch, training=True)
        disc_h3_fake1, disc_h3_fake2, disc_h3_fake3 = D_h3(h3_fake3, training=True)
        
        # obtain reduced batches
        h4_batch_2 = tf.nn.pool(h4_batch, window_shape=(2, 2, 2), pooling_type='AVG', strides=(2, 2, 2), padding='VALID')
        h4_batch_1 = tf.nn.pool(h4_batch_2, window_shape=(2, 2, 2), pooling_type='AVG', strides=(2, 2, 2), padding='VALID')
        
        h3_batch_2 = tf.nn.pool(h3_batch, window_shape=(2, 2, 2), pooling_type='AVG', strides=(2, 2, 2), padding='VALID')
        h3_batch_1 = tf.nn.pool(h3_batch_2, window_shape=(2, 2, 2), pooling_type='AVG', strides=(2, 2, 2), padding='VALID')

        # obtain reduced masks batches
        h4_batch_bmask_2 = tf.nn.pool(h4_batch_bmask, window_shape=(2, 2, 2), pooling_type='AVG', strides=(2, 2, 2), padding='VALID')
        h4_batch_bmask_1 = tf.nn.pool(h4_batch_bmask_2, window_shape=(2, 2, 2), pooling_type='AVG', strides=(2, 2, 2), padding='VALID')
        
        h3_batch_bmask_2 = tf.nn.pool(h3_batch_bmask, window_shape=(2, 2, 2), pooling_type='AVG', strides=(2, 2, 2), padding='VALID')
        h3_batch_bmask_1 = tf.nn.pool(h3_batch_bmask_2, window_shape=(2, 2, 2), pooling_type='AVG', strides=(2, 2, 2), padding='VALID')

        # ---- apply masks

        # original images
        h4_batch *= h4_batch_bmask
        h4_batch_2 *= h4_batch_bmask_2
        h4_batch_1 *= h4_batch_bmask_1

        h3_batch *= h3_batch_bmask
        h3_batch_2 *= h3_batch_bmask_2
        h3_batch_1 *= h3_batch_bmask_1

        # fake images
        h4_fake3 *= h4_batch_bmask
        h4_fake2 *= h4_batch_bmask_2
        h4_fake1 *= h4_batch_bmask_1

        h3_fake3 *= h3_batch_bmask
        h3_fake2 *= h3_batch_bmask_2
        h3_fake1 *= h3_batch_bmask_1

        # cycled images
        h4_fake3_cyc *= h4_batch_bmask
        h4_fake2_cyc *= h4_batch_bmask_2
        h4_fake1_cyc *= h4_batch_bmask_1

        h3_fake3_cyc *= h3_batch_bmask
        h3_fake2_cyc *= h3_batch_bmask_2
        h3_fake1_cyc *= h3_batch_bmask_1

        # discriminator real
        disc_h4_real3 *= h4_batch_bmask
        disc_h4_real2 *= h4_batch_bmask_2
        disc_h4_real1 *= h4_batch_bmask_1

        disc_h3_real3 *= h3_batch_bmask
        disc_h3_real2 *= h3_batch_bmask_2
        disc_h3_real1 *= h3_batch_bmask_1

        # discriminator fake
        disc_h4_fake3 *= h4_batch_bmask
        disc_h4_fake2 *= h4_batch_bmask_2
        disc_h4_fake1 *= h4_batch_bmask_1

        disc_h3_fake3 *= h3_batch_bmask
        disc_h3_fake2 *= h3_batch_bmask_2
        disc_h3_fake1 *= h3_batch_bmask_1
        

        # ---- Adversarial loss
        ls1_adv = adversarial_loss(disc_h4_real1, disc_h3_real1, disc_h4_fake1, disc_h3_fake1)
        ls2_adv = adversarial_loss(disc_h4_real2, disc_h3_real2, disc_h4_fake2, disc_h3_fake2)
        ls3_adv = adversarial_loss(disc_h4_real3, disc_h3_real3, disc_h4_fake3, disc_h3_fake3)

        # ---- Paired loss
        # Quality loss
        ls1_qa_g_h4, ls1_qa_g_h3 = quality_loss(h4_batch_1, h4_fake1, disc_h4_fake1), quality_loss(h3_batch_1, h3_fake1, disc_h3_fake1)
        ls2_qa_g_h4, ls2_qa_g_h3 = quality_loss(h4_batch_2, h4_fake2, disc_h4_fake2), quality_loss(h3_batch_2, h3_fake2, disc_h3_fake2)
        ls3_qa_g_h4, ls3_qa_g_h3 = quality_loss(h4_batch, h4_fake3, disc_h4_fake3), quality_loss(h3_batch, h3_fake3, disc_h3_fake3)

        # Texture loss
        ls1_tex_g_h4, ls1_tex_g_h3 = texture_loss(h4_batch_1, h4_fake1), texture_loss(h3_batch_1, h3_fake1)
        ls2_tex_g_h4, ls2_tex_g_h3 = texture_loss(h4_batch_2, h4_fake2), texture_loss(h3_batch_2, h3_fake2)
        ls3_tex_g_h4, ls3_tex_g_h3 = texture_loss(h4_batch, h4_fake3), texture_loss(h3_batch, h3_fake3)

        # Frequency loss
        ls1_freq_g_h4, ls1_freq_g_h3 = frequency_loss(h4_batch_1, h4_fake1), frequency_loss(h3_batch_1, h3_fake1)
        ls2_freq_g_h4, ls2_freq_g_h3 = frequency_loss(h4_batch_2, h4_fake2), frequency_loss(h3_batch_2, h3_fake2)
        ls3_freq_g_h4, ls3_freq_g_h3 = frequency_loss(h4_batch, h4_fake3), frequency_loss(h3_batch, h3_fake3)

        # ---- Cycle consistency loss
        lsc1 = calc_cycle_loss(h4_batch_1, h4_fake1_cyc) + calc_cycle_loss(h3_batch_1, h3_fake1_cyc)
        lsc2 = calc_cycle_loss(h4_batch_2, h4_fake2_cyc) + calc_cycle_loss(h3_batch_2, h3_fake2_cyc)
        lsc3 = calc_cycle_loss(h4_batch, h4_fake3_cyc) + calc_cycle_loss(h3_batch, h3_fake3_cyc)

        # ---- Total loss
        ls1_g_h4 = ls1_adv + ls1_qa_g_h4 + ls1_tex_g_h4 + ls1_freq_g_h4 + lsc1
        ls2_g_h4 = ls2_adv + ls2_qa_g_h4 + ls2_tex_g_h4 + ls2_freq_g_h4 + lsc2
        ls3_g_h4 = ls3_adv + ls3_qa_g_h4 + ls3_tex_g_h4 + ls3_freq_g_h4 + lsc3

        ls1_g_h3 = ls1_adv + ls1_qa_g_h3 + ls1_tex_g_h3 + ls1_freq_g_h3 + lsc1
        ls2_g_h3 = ls2_adv + ls2_qa_g_h3 + ls2_tex_g_h3 + ls2_freq_g_h3 + lsc2
        ls3_g_h3 = ls3_adv + ls3_qa_g_h3 + ls3_tex_g_h3 + ls3_freq_g_h3 + lsc3

        # ---- generator losses
        g_h4_loss = ls1_g_h4 + ls2_g_h4 + ls3_g_h4
        g_h3_loss = ls1_g_h3 + ls2_g_h3 + ls3_g_h3

        # ---- discriminator losses
        d_loss = ls1_adv + ls2_adv + ls3_adv


    g_h4_gradients = g_h4_tape.gradient(g_h4_loss, G_h4.trainable_variables)
    g_h3_gradients = g_h3_tape.gradient(g_h3_loss, G_h3.trainable_variables)

    d_h4_gradients = d_h4_tape.gradient(d_loss, D_h4.trainable_variables)
    d_h3_gradients = d_h3_tape.gradient(d_loss, D_h3.trainable_variables)

    # ---- apply gradients
    g_h4_optimizer.apply_gradients(zip(g_h4_gradients, G_h4.trainable_variables))
    g_h3_optimizer.apply_gradients(zip(g_h3_gradients, G_h3.trainable_variables))

    d_h4_optimizer.apply_gradients(zip(d_h4_gradients, D_h4.trainable_variables))
    d_h3_optimizer.apply_gradients(zip(d_h3_gradients, D_h3.trainable_variables))

    return d_loss, g_h4_loss, g_h3_loss















def train(num_epochs, batch_size, monitor_each=100, show=False):
    image_size = (193,229,193)
    patch_size= (64,64,64)
    patch_step = (32,32,32)

    # ---- Models
    D_h4 = Discriminator.MetamorphicDiscriminator()
    G_h4 = Generator.MetamorphicGenerator()

    D_h3 = Discriminator.MetamorphicDiscriminator()
    G_h3 = Generator.MetamorphicGenerator()

    # optimizers
    inital_lr = 1e-5
    g_h3_optimizer = tf.keras.optimizers.Adam(inital_lr)
    d_h3_optimizer = tf.keras.optimizers.Adam(inital_lr)

    g_h4_optimizer = tf.keras.optimizers.Adam(inital_lr)
    d_h4_optimizer = tf.keras.optimizers.Adam(inital_lr)
    
    g_h3_optimizer_monitor = udl.LearningRateMonitor(g_h3_optimizer, inital_lr, 0.5, 1e-10, 20, "loss")
    g_h4_optimizer_monitor = udl.LearningRateMonitor(g_h4_optimizer, inital_lr, 0.5, 1e-10, 20, "loss")
    d_h3_optimizer_monitor = udl.LearningRateMonitor(d_h3_optimizer, inital_lr, 0.5, 1e-10, 20, "loss")
    d_h4_optimizer_monitor = udl.LearningRateMonitor(d_h4_optimizer, inital_lr, 0.5, 1e-10, 20, "loss")

    # ---- Data
    server = True
    data_train = DataLoaderPatches(pil.INAME_PREPROCESSED,
                                   batch_size=batch_size, 
                                   image_size=image_size, 
                                   patch_size=patch_size,
                                   patch_step=patch_step,
                                   images_onmemory=1,
                                   shuffle_patches=False,
                                   channels_first=False,
                                   background_th=0.15,
                                   shuffle_images=True,
                                   nb_patients=620,
                                    shuffle_before_selecting=False,
                                    server=server,
                                    exclude_ids=["08202"],
                                    data_type=np.float32)
    
    data_train.prepare_custom_patches("bmask", pil.INAME_BRAIN_MASK.format("{0}", "{1}", pil.REGISTRATION_TO_MNI))
    data_train.prepare_custom_patches("seg3", pil.INAME_SEGMENTATION.format("{0}", "{1}", 3, "nii.gz"))
    weights={"bg":0, "ext":0.5, "csf":2, "gm":3, "wm":1}

    # visualization
    nb_loss = 3
    acum_loss_train = udl.AcumulativeMetrics(nb_metrics = nb_loss)

    log_dir=f"{current_path}/models/logs/train"
    summary_writer = tf.summary.create_file_writer(log_dir)


    # ---- checkpoint
    checkpoint_dir = f"{current_path}/models/training_checkpoints/"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(g_h3_optimizer=g_h3_optimizer,
                                     g_h4_optimizer=g_h4_optimizer,
                                    d_h3_optimizer=d_h3_optimizer,
                                    d_h4_optimizer=d_h4_optimizer,
                                    D_h3=D_h3,
                                    D_h4=D_h4,
                                    G_h3=G_h3,
                                    G_h4=G_h4)
    
    # checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir)
    # restore checkpoint
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    for epoch in range(0,num_epochs):

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # visualize images
        if epoch == 0:
            show_generated_image("08202", G_h4, G_h3, patch_size, patch_step, current_plane=116, show=show)
        
        # ---- Train
        acum_loss_train.reset()
        progress_bar = tqdm(total=len(data_train), desc=f'Initializing...')
        for step, (h3_batch, h4_batch) in enumerate(data_train):
            # _, bmask_hunt4 = data_train.get_iter_custom_patches("bmask")
            bmask_hunt3, bmask_hunt4 = get_weight_mask_hunt4(data_train, weights)
            d_loss, g_h4_loss, g_h3_loss = train_step(G_h3, D_h3, G_h4, D_h4, h3_batch, h4_batch, g_h3_optimizer, d_h3_optimizer, g_h4_optimizer, d_h4_optimizer, bmask_hunt3, bmask_hunt4)
            acum_loss_train.update((d_loss, g_h4_loss, g_h3_loss))
            loss_mean = acum_loss_train.get_mean()
            
            progress_bar.set_description(desc=f'd_loss: {loss_mean[0]:.5f} \t g_h4_loss: {loss_mean[1]:.5f} \t g_h3_loss: {loss_mean[2]:.5f} ')
            progress_bar.update(1)
        del progress_bar

            
        loss_mean = acum_loss_train.get_mean()

        g_h3_optimizer_monitor.on_epoch_end(loss_mean[2])
        g_h4_optimizer_monitor.on_epoch_end(loss_mean[1])
        d_h3_optimizer_monitor.on_epoch_end(loss_mean[0])
        d_h4_optimizer_monitor.on_epoch_end(loss_mean[0])


        if (epoch + 1) % monitor_each == 0:
            # ---- Check points
            checkpoint.save(file_prefix=checkpoint_prefix)

            # ---- visuqlize
            if show:
                show_generated_image("00046", G_h4, G_h3, patch_size, patch_step, current_plane=116, show=show)
            val_results = show_generated_image("08202", G_h4, G_h3, patch_size, patch_step, current_plane=116, show=show)

            # ---- tensorboard
            with summary_writer.as_default():
                tf.summary.scalar('d_loss', loss_mean[0], step=epoch)
                tf.summary.scalar('g_h4_loss', loss_mean[1], step=epoch)
                tf.summary.scalar('g_h3_loss', loss_mean[2], step=epoch)
                
                # images
                gen_image_h4 = np.expand_dims(np.expand_dims(val_results[2], axis=-1), axis=0)
                hunt4_vs_gen_h4 = np.expand_dims(np.stack([val_results[2], val_results[1], np.zeros_like(val_results[1])], axis=-1), axis=0)
                tf.summary.image("generated H4", gen_image_h4, step=epoch)
                tf.summary.image("R: Gen H4 - G: HUNT4", hunt4_vs_gen_h4, step=epoch)
        
                gen_image_h3 = np.expand_dims(np.expand_dims(val_results[3], axis=-1), axis=0)
                hunt3_vs_gen_h3 = np.expand_dims(np.stack([val_results[3], val_results[0], np.zeros_like(val_results[0])], axis=-1), axis=0)
                tf.summary.image("generated H3", gen_image_h3, step=epoch)
                tf.summary.image("R: Gen H3 - G: HUNT3", hunt3_vs_gen_h3, step=epoch)

train(num_epochs=100000, batch_size=1, monitor_each=20, show=False)