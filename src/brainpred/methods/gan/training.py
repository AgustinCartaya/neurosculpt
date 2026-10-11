"""Training loops of the GAN variants (MGAN: patch-based cycle GAN; MPGAN: whole-volume pix2pix).

Use :func:`train` (called by ``scripts/train_gan.py``). Checkpoints are written every
``variant.monitor_each`` epochs to ``<checkpoints_dir>/<method_name>/training_checkpoints/ckpt-N`` and
training resumes from the latest one. For MGAN one "epoch" is one pass over the patches of a single
patient (the loader moves to the next patient after each pass), as in the original code.
"""
import datetime
import os

import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom
from tqdm import tqdm

from brainpred import paths, splits
from brainpred.methods.gan import losses as L
from brainpred.methods.gan import variants as V
from brainpred.methods.gan.inference import generate_image_full, generate_image_patches
from brainpred.utils import deep_learning as udl
from brainpred.utils import segmentation as ufs

LR_REDUCE_FACTOR = 0.5
LR_MIN = 1e-10
PREVIEW_PLANE = 116
SEG_3_SUFIX = paths.INAME_SEGMENTATION.format("{0}", "{1}", 3, "nii.gz")


# ---------------------------------------------------------------- helpers
def _avg_pool_2(x):
    return tf.nn.pool(x, window_shape=(2, 2, 2), pooling_type='AVG', strides=(2, 2, 2), padding='VALID')


def reduce_3D_images_batch(img_batch, scaling_factor=2, channels_first=False):
    """Nearest-neighbour downsampling of a batch of 3D images (port of OLD utils/functions.py)."""
    if not channels_first:
        img_batch = np.transpose(img_batch, (0, 4, 1, 2, 3))

    new_shape = np.array(img_batch.shape[2:]) // scaling_factor
    img_batch_resized = np.zeros((*img_batch.shape[:2], *new_shape), dtype=img_batch.dtype)
    for i in range(img_batch.shape[0]):
        for c in range(img_batch.shape[1]):
            img_batch_resized[i, c] = zoom(img_batch[i, c], zoom=1 / scaling_factor, order=0)

    if not channels_first:
        img_batch_resized = np.transpose(img_batch_resized, (0, 2, 3, 4, 1))
    return img_batch_resized


def get_2_scales_reduction(img):
    img_red = reduce_3D_images_batch(img, scaling_factor=2)
    img_red_red = reduce_3D_images_batch(img_red, scaling_factor=2)
    return img_red, img_red_red


def _apply_gradients(tape, loss, model, optimizer):
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def default_train_ids():
    """Training patients: the train split minus the extra validation subjects.

    Equivalent to the original ``nb_patients=620, exclude_ids=["08202"]`` selection.
    """
    validation_ids = set(splits.get_validation_ids())
    return [i for i in splits.get_train_ids() if i not in validation_ids]


def _write_tensorboard_images(summary_writer, epoch, loss_names, loss_mean, h3, h4, gen_h4, gen_h3=None):
    with summary_writer.as_default():
        for name, value in zip(loss_names, loss_mean):
            tf.summary.scalar(name, value, step=epoch)

        gen_image_h4 = np.expand_dims(np.expand_dims(gen_h4, axis=-1), axis=0)
        hunt4_vs_gen_h4 = np.expand_dims(np.stack([gen_h4, h4, np.zeros_like(h4)], axis=-1), axis=0)
        tf.summary.image("generated H4", gen_image_h4, step=epoch)
        tf.summary.image("R: Gen H4 - G: HUNT4", hunt4_vs_gen_h4, step=epoch)

        if gen_h3 is not None:
            gen_image_h3 = np.expand_dims(np.expand_dims(gen_h3, axis=-1), axis=0)
            hunt3_vs_gen_h3 = np.expand_dims(np.stack([gen_h3, h3, np.zeros_like(h3)], axis=-1), axis=0)
            tf.summary.image("generated H3", gen_image_h3, step=epoch)
            tf.summary.image("R: Gen H3 - G: HUNT3", hunt3_vs_gen_h3, step=epoch)


def _make_summary_writer(variant):
    log_dir = f"{V.get_log_dir(variant)}/train" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return tf.summary.create_file_writer(log_dir)


def _restore_latest(checkpoint, checkpoint_dir):
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print(f"Restoring checkpoint: {latest}")
    checkpoint.restore(latest)


# ---------------------------------------------------------------- MGAN
def make_mgan_train_steps(G_h3, D_h3, G_h4, D_h4, g_h3_optimizer, d_h3_optimizer, g_h4_optimizer, d_h4_optimizer,
                          dwt_layer):
    """Return ``(train_step, train_step_seg)`` tf.functions for MGAN_paper_16 and MGAN_paper_16_seg_loss.

    Both run the two generators (HUNT3->HUNT4 and HUNT4->HUNT3) and discriminators with 3-scale deep
    supervision and the losses: adversarial + quality + texture + frequency + cycle (+ segmentation),
    all with weight 1. They differ in how the lower scales of the targets are obtained (average pooling
    in-graph vs. nearest-neighbour precomputed scales) and in the seg-loss variant masking images,
    cycles and discriminator outputs with the segmentation before computing the losses.
    """

    def forward(h3_batch, h4_batch):
        h4_fake = G_h4(h3_batch, training=True)
        h3_cyc = G_h3(h4_fake[2], training=True)

        h3_fake = G_h3(h4_batch, training=True)
        h4_cyc = G_h4(h3_fake[2], training=True)

        disc_h4_real = D_h4(h4_batch, training=True)
        disc_h4_fake = D_h4(h4_fake[2], training=True)
        disc_h3_real = D_h3(h3_batch, training=True)
        disc_h3_fake = D_h3(h3_fake[2], training=True)
        return [list(t) for t in (h4_fake, h3_cyc, h3_fake, h4_cyc, disc_h4_real, disc_h4_fake, disc_h3_real,
                                  disc_h3_fake)]

    def compute_losses(h3_s, h4_s, h4_fake, h3_cyc, h3_fake, h4_cyc, disc_h4_real, disc_h4_fake, disc_h3_real,
                       disc_h3_fake, seg_h4=None, seg_h3=None):
        """All arguments are lists over the 3 scales (16^3, 32^3, 64^3)."""
        g_h4_loss, g_h3_loss, d_loss = 0.0, 0.0, 0.0
        for s in range(3):
            adv = L.adversarial_loss(disc_h4_real[s], disc_h3_real[s], disc_h4_fake[s], disc_h3_fake[s])
            qa_h4 = L.quality_loss(h4_s[s], h4_fake[s], disc_h4_fake[s])
            qa_h3 = L.quality_loss(h3_s[s], h3_fake[s], disc_h3_fake[s])
            tex_h4 = L.texture_loss(h4_s[s], h4_fake[s])
            tex_h3 = L.texture_loss(h3_s[s], h3_fake[s])
            freq_h4 = L.frequency_loss(h4_s[s], h4_fake[s], dwt_layer)
            freq_h3 = L.frequency_loss(h3_s[s], h3_fake[s], dwt_layer)
            lsc = L.cycle_loss(h4_s[s], h4_cyc[s]) + L.cycle_loss(h3_s[s], h3_cyc[s])

            ls_g_h4 = adv + qa_h4 + tex_h4 + freq_h4 + lsc
            ls_g_h3 = adv + qa_h3 + tex_h3 + freq_h3 + lsc
            if seg_h4 is not None:
                ls_g_h4 = ls_g_h4 + seg_h4[s]
                ls_g_h3 = ls_g_h3 + seg_h3[s]

            g_h4_loss += ls_g_h4
            g_h3_loss += ls_g_h3
            d_loss += adv
        return g_h4_loss, g_h3_loss, d_loss

    def apply_all(tapes, g_h4_loss, g_h3_loss, d_loss):
        g_h4_tape, d_h4_tape, g_h3_tape, d_h3_tape = tapes
        _apply_gradients(g_h4_tape, g_h4_loss, G_h4, g_h4_optimizer)
        _apply_gradients(g_h3_tape, g_h3_loss, G_h3, g_h3_optimizer)
        _apply_gradients(d_h4_tape, d_loss, D_h4, d_h4_optimizer)
        _apply_gradients(d_h3_tape, d_loss, D_h3, d_h3_optimizer)

    @tf.function
    def train_step(h3_batch, h4_batch):
        with tf.GradientTape() as g_h4_tape, tf.GradientTape() as d_h4_tape, \
                tf.GradientTape() as g_h3_tape, tf.GradientTape() as d_h3_tape:
            outputs = forward(h3_batch, h4_batch)

            h4_batch_2 = _avg_pool_2(h4_batch)
            h4_batch_1 = _avg_pool_2(h4_batch_2)
            h3_batch_2 = _avg_pool_2(h3_batch)
            h3_batch_1 = _avg_pool_2(h3_batch_2)

            g_h4_loss, g_h3_loss, d_loss = compute_losses([h3_batch_1, h3_batch_2, h3_batch],
                                                          [h4_batch_1, h4_batch_2, h4_batch], *outputs)

        apply_all((g_h4_tape, d_h4_tape, g_h3_tape, d_h3_tape), g_h4_loss, g_h3_loss, d_loss)
        return d_loss, g_h4_loss, g_h3_loss

    @tf.function
    def train_step_seg(h3_batch, h4_batch, h3_seg, h4_seg, h3_gmm, h4_gmm):
        """``h3_batch``, ``h4_batch``, ``h3_seg``, ``h4_seg``: tuples of the 3 scales (16^3, 32^3, 64^3)."""
        with tf.GradientTape() as g_h4_tape, tf.GradientTape() as d_h4_tape, \
                tf.GradientTape() as g_h3_tape, tf.GradientTape() as d_h3_tape:
            outputs = forward(h3_batch[2], h4_batch[2])

            # mask targets and every generated / discriminator output with the segmentation
            h3_s = [L.apply_mask(h3_batch[s], h3_seg[s]) for s in range(3)]
            h4_s = [L.apply_mask(h4_batch[s], h4_seg[s]) for s in range(3)]
            seg_order = (h4_seg, h3_seg, h3_seg, h4_seg, h4_seg, h4_seg, h3_seg, h3_seg)
            outputs = [[L.apply_mask(out[s], seg[s]) for s in range(3)] for out, seg in zip(outputs, seg_order)]
            h4_fake, h3_fake = outputs[0], outputs[2]

            seg_h4 = [L.segmentation_dice_loss(h4_seg[s], h4_fake[s], h4_gmm[0], h4_gmm[1], h4_gmm[2], h4_gmm[3])
                      for s in range(3)]
            seg_h3 = [L.segmentation_dice_loss(h3_seg[s], h3_fake[s], h3_gmm[0], h3_gmm[1], h3_gmm[2], h3_gmm[3])
                      for s in range(3)]

            g_h4_loss, g_h3_loss, d_loss = compute_losses(h3_s, h4_s, *outputs, seg_h4=seg_h4, seg_h3=seg_h3)

        apply_all((g_h4_tape, d_h4_tape, g_h3_tape, d_h3_tape), g_h4_loss, g_h3_loss, d_loss)
        return d_loss, g_h4_loss, g_h3_loss

    return train_step, train_step_seg


def _mgan_preview(id, G_h4, G_h3, show):
    from brainpred.utils import display

    img_hunt3, _ = paths.load_preprocessed_image(3, id)
    img_hunt4, _ = paths.load_preprocessed_image(4, id)
    img_gen_h4 = generate_image_patches(img_hunt3, G_h4)
    img_gen_h3 = generate_image_patches(img_hunt4, G_h3)

    _, _, img_gen_h4 = display.shows_3_images(img_hunt3, img_hunt4, img_gen_h4, current_plane=PREVIEW_PLANE, show=show)
    img_hunt3, img_hunt4, img_gen_h3 = display.shows_3_images(img_hunt3, img_hunt4, img_gen_h3,
                                                              current_plane=PREVIEW_PLANE, show=show)
    return img_hunt3, img_hunt4, img_gen_h4, img_gen_h3


def train_mgan(variant, batch_size=1, num_epochs=None, monitor_each=None, show=False, preview_id=None,
               train_ids=None):
    from brainpred.data.patches import DataLoaderPatches
    from brainpred.methods.gan.layers.wavelet3d import DWT3D
    from brainpred.methods.gan.mgan.discriminator import MetamorphicDiscriminator
    from brainpred.methods.gan.mgan.generator import MetamorphicGenerator

    num_epochs = num_epochs or variant.num_epochs
    monitor_each = monitor_each or variant.monitor_each
    preview_id = preview_id or splits.get_validation_ids()[0]

    # ---- Models
    D_h4, G_h4 = MetamorphicDiscriminator(norm=variant.norm), MetamorphicGenerator(norm=variant.norm)
    D_h3, G_h3 = MetamorphicDiscriminator(norm=variant.norm), MetamorphicGenerator(norm=variant.norm)
    dwt_layer = DWT3D(wavelet_name="bior1.3")

    # ---- Optimizers
    lr = variant.learning_rate
    g_h3_optimizer, d_h3_optimizer = tf.keras.optimizers.Adam(lr), tf.keras.optimizers.Adam(lr)
    g_h4_optimizer, d_h4_optimizer = tf.keras.optimizers.Adam(lr), tf.keras.optimizers.Adam(lr)
    monitor = lambda opt: udl.LearningRateMonitor(opt, lr, LR_REDUCE_FACTOR, LR_MIN, variant.lr_patience, "loss")
    g_h3_monitor, g_h4_monitor = monitor(g_h3_optimizer), monitor(g_h4_optimizer)
    d_h3_monitor, d_h4_monitor = monitor(d_h3_optimizer), monitor(d_h4_optimizer)

    train_step, train_step_seg = make_mgan_train_steps(G_h3, D_h3, G_h4, D_h4, g_h3_optimizer, d_h3_optimizer,
                                                       g_h4_optimizer, d_h4_optimizer, dwt_layer)

    # ---- Data
    data_train = DataLoaderPatches(paths.INAME_PREPROCESSED,
                                   batch_size=batch_size,
                                   image_size=V.IMAGE_SIZE,
                                   patch_size=V.MGAN_PATCH_SIZE,
                                   patch_step=V.MGAN_PATCH_STEP,
                                   images_onmemory=1,
                                   shuffle_patches=variant.shuffle_patches,
                                   channels_first=False,
                                   background_th=V.MGAN_BACKGROUND_TH,
                                   shuffle_images=True,
                                   patient_ids=train_ids or default_train_ids(),
                                   data_type=np.float32,
                                   image_sufix_seg_gmm=SEG_3_SUFIX if variant.seg_loss else None)
    if variant.seg_loss:
        data_train.prepare_custom_patches("seg3", SEG_3_SUFIX, data_type=np.int32)

    loss_names = ['d_loss', 'g_h4_loss', 'g_h3_loss']
    acum_loss_train = udl.AcumulativeMetrics(nb_metrics=len(loss_names))
    summary_writer = _make_summary_writer(variant)

    # ---- Checkpoint (attribute names must match the original checkpoints)
    checkpoint_dir = V.get_checkpoint_dir(variant)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(g_h3_optimizer=g_h3_optimizer,
                                     g_h4_optimizer=g_h4_optimizer,
                                     d_h3_optimizer=d_h3_optimizer,
                                     d_h4_optimizer=d_h4_optimizer,
                                     D_h3=D_h3,
                                     D_h4=D_h4,
                                     G_h3=G_h3,
                                     G_h4=G_h4)
    _restore_latest(checkpoint, checkpoint_dir)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        if epoch == 0 and show:
            _mgan_preview(preview_id, G_h4, G_h3, show)

        # ---- Train
        acum_loss_train.reset()
        progress_bar = tqdm(total=len(data_train), desc='Initializing...')
        for h3_batch, h4_batch in data_train:
            if variant.seg_loss:
                h3_seg_3, h4_seg_3 = data_train.get_iter_custom_patches("seg3")
                h3_gmm, h4_gmm = data_train.get_iter_patient_gmm()
                h3_batch_2, h3_batch_1 = get_2_scales_reduction(h3_batch)
                h4_batch_2, h4_batch_1 = get_2_scales_reduction(h4_batch)
                h3_seg_2, h3_seg_1 = get_2_scales_reduction(h3_seg_3)
                h4_seg_2, h4_seg_1 = get_2_scales_reduction(h4_seg_3)
                step_losses = train_step_seg((h3_batch_1, h3_batch_2, h3_batch),
                                             (h4_batch_1, h4_batch_2, h4_batch),
                                             (h3_seg_1, h3_seg_2, h3_seg_3),
                                             (h4_seg_1, h4_seg_2, h4_seg_3),
                                             h3_gmm, h4_gmm)
            else:
                step_losses = train_step(h3_batch, h4_batch)

            acum_loss_train.update(step_losses)
            progress_bar.set_description(desc=acum_loss_train.get_progress_bar_text(loss_names))
            progress_bar.update(1)
        progress_bar.close()

        loss_mean = acum_loss_train.get_mean()
        g_h3_monitor.on_epoch_end(loss_mean[2])
        g_h4_monitor.on_epoch_end(loss_mean[1])
        d_h3_monitor.on_epoch_end(loss_mean[0])
        d_h4_monitor.on_epoch_end(loss_mean[0])

        if (epoch + 1) % monitor_each == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            h3, h4, gen_h4, gen_h3 = _mgan_preview(preview_id, G_h4, G_h3, show)
            _write_tensorboard_images(summary_writer, epoch, loss_names, loss_mean, h3, h4, gen_h4, gen_h3)


# ---------------------------------------------------------------- MPGAN
def make_mpgan_train_step(G, D, Fe, g_optimizer, d_optimizer, seg_loss):
    """Return the tf.function training step of MPGAN_paper_16 (``seg_loss=False``) or MPGAN_paper_16_seg_loss.

    * paper:    adv + 25 * L1(foreground of real) + 25 * perceptual
    * seg loss: adv + 25 * L1(inside segmentation) + 20 * perceptual + 15 * GMM soft Dice
    """

    @tf.function
    def train_step(h3_batch, h4_batch, seg_hunt4=None, gmm=None):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            h4_fake = G(h3_batch, training=True)

            real_output = D(h4_batch, training=True)
            fake_output = D(h4_fake, training=True)

            adv_gen_loss = L.generator_loss(fake_output)
            disc_loss = L.discriminator_loss(real_output, fake_output)

            if seg_loss:
                l1_loss = L.voxel_loss(h4_batch, h4_fake, seg_hunt4)
                perceptual_loss = L.feature_extractor_loss(Fe, h4_batch, h4_fake)
                seg_dice_loss = L.segmentation_dice_loss(seg_hunt4, h4_fake, *gmm)
                gen_loss = adv_gen_loss + 25.0 * l1_loss + 20.0 * perceptual_loss + 15.0 * seg_dice_loss
            else:
                l1_loss = L.foreground_l1_loss(h4_batch, h4_fake)
                perceptual_loss = L.feature_extractor_loss(Fe, h4_batch, h4_fake)
                gen_loss = adv_gen_loss + 25.0 * l1_loss + 25.0 * perceptual_loss

        _apply_gradients(gen_tape, gen_loss, G, g_optimizer)
        _apply_gradients(disc_tape, disc_loss, D, d_optimizer)

        if seg_loss:
            return gen_loss, disc_loss, l1_loss, perceptual_loss, seg_dice_loss
        return gen_loss, disc_loss, l1_loss, perceptual_loss

    return train_step


def _mpgan_preview(id, G, show):
    from brainpred.utils import display

    img_hunt3 = paths.load_preprocessed_image(3, id)[0]
    img_hunt4 = paths.load_preprocessed_image(4, id)[0]
    img_gen_h4 = generate_image_full(img_hunt3, G)
    return display.shows_3_images(img_hunt3, img_hunt4, img_gen_h4, current_plane=PREVIEW_PLANE, show=show)


def train_mpgan(variant, batch_size=1, num_epochs=None, monitor_each=None, show=False, preview_id=None,
                train_ids=None):
    from brainpred.data.complete import DataLoaderComplete
    from brainpred.methods.gan.mpgan.discriminator import create_discriminator_model
    from brainpred.methods.gan.mpgan.feature_extractor import create_features_extractor
    from brainpred.methods.gan.mpgan.generator import create_generator_model

    num_epochs = num_epochs or variant.num_epochs
    monitor_each = monitor_each or variant.monitor_each
    preview_id = preview_id or splits.get_validation_ids()[0]
    input_shape = V.MPGAN_INPUT_SHAPE

    # ---- Models (the feature extractor is pretrained and never updated)
    D = create_discriminator_model(input_shape, norm=variant.norm)
    G = create_generator_model(input_shape, norm=variant.norm)
    Fe = create_features_extractor((1,) + input_shape, paths.get_genesis_weights_path_name())

    # ---- Optimizers
    lr = variant.learning_rate
    g_optimizer, d_optimizer = tf.keras.optimizers.Adam(lr), tf.keras.optimizers.Adam(lr)
    g_monitor = udl.LearningRateMonitor(g_optimizer, lr, LR_REDUCE_FACTOR, LR_MIN, variant.lr_patience, "loss")
    d_monitor = udl.LearningRateMonitor(d_optimizer, lr, LR_REDUCE_FACTOR, LR_MIN, variant.lr_patience, "loss")

    train_step = make_mpgan_train_step(G, D, Fe, g_optimizer, d_optimizer, variant.seg_loss)

    # ---- Data
    data_train = DataLoaderComplete(image_sufix=paths.INAME_PREPROCESSED,
                                    batch_size=batch_size,
                                    image_size=input_shape,
                                    channels_first=True,
                                    shuffle=True,
                                    patient_ids=train_ids or default_train_ids(),
                                    data_type=np.float32,
                                    img_size_offset=V.MPGAN_IMG_OFFSET)

    loss_names = ['gen_loss', 'disc_loss', 'l1_loss', 'percetual_loss']
    if variant.seg_loss:
        loss_names.append('seg_dice_loss')
    acum_loss_train = udl.AcumulativeMetrics(nb_metrics=len(loss_names))
    summary_writer = _make_summary_writer(variant)

    # ---- Checkpoint (attribute names must match the original checkpoints)
    checkpoint_dir = V.get_checkpoint_dir(variant)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(g_optimizer=g_optimizer, d_optimizer=d_optimizer, G=G, D=D)
    _restore_latest(checkpoint, checkpoint_dir)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        if epoch == 0 and show:
            _mpgan_preview(preview_id, G, show)

        # ---- Train
        acum_loss_train.reset()
        progress_bar = tqdm(total=len(data_train), desc='Initializing...')
        for h3_batch, h4_batch in data_train:
            if variant.seg_loss:
                _, seg_hunt4 = data_train.get_iter_custom_images(SEG_3_SUFIX, data_type=np.int8)
                gmm = ufs.obtain_multiple_gmm_elements_using_seg(h4_batch[:, 0], seg_hunt4[:, 0])
                step_losses = train_step(h3_batch, h4_batch, seg_hunt4, gmm)
            else:
                step_losses = train_step(h3_batch, h4_batch)

            acum_loss_train.update(step_losses)
            progress_bar.set_description(desc=acum_loss_train.get_progress_bar_text(loss_names))
            progress_bar.update(1)
        progress_bar.close()

        loss_mean = acum_loss_train.get_mean()
        g_monitor.on_epoch_end(loss_mean[0])
        d_monitor.on_epoch_end(loss_mean[1])

        if (epoch + 1) % monitor_each == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            h3, h4, gen_h4 = _mpgan_preview(preview_id, G, show)
            _write_tensorboard_images(summary_writer, epoch, loss_names, loss_mean, h3, h4, gen_h4)


# ---------------------------------------------------------------- entry point
def train(variant, **kwargs):
    """Configure TensorFlow and train ``variant`` (a :class:`~brainpred.methods.gan.variants.GanVariant`)."""
    V.configure_tensorflow(variant)
    if variant.family == V.FAMILY_MGAN:
        return train_mgan(variant, **kwargs)
    return train_mpgan(variant, **kwargs)
