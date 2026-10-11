"""Loss functions of the MGAN and MPGAN variants."""
import tensorflow as tf

from brainpred.utils import deep_learning as udl

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)


# ---------------------------------------------------------------- MGAN (cycle GAN, two directions)
def adversarial_loss(disc_h4_real, disc_h3_real, disc_h4_fake, disc_h3_fake):
    """Binary cross-entropy of both discriminators on real and fake inputs (shared by G and D)."""
    real_loss_h3 = cross_entropy(tf.ones_like(disc_h3_real), disc_h3_real)
    real_loss_h4 = cross_entropy(tf.ones_like(disc_h4_real), disc_h4_real)
    fake_loss_h3 = cross_entropy(tf.zeros_like(disc_h3_fake), disc_h3_fake)
    fake_loss_h4 = cross_entropy(tf.zeros_like(disc_h4_fake), disc_h4_fake)
    return real_loss_h3 + real_loss_h4 + fake_loss_h3 + fake_loss_h4


def quality_loss(target_image, predicted_image, disc_prob, beta=1.5):
    """L1 weighted by how confidently the discriminator rejects each voxel."""
    return tf.reduce_mean(tf.abs(target_image - predicted_image) * tf.pow(1 - disc_prob, beta))


def gram_matrix(input_tensor):
    batch_size, width, height, deep, channels = input_tensor.get_shape().as_list()
    reshaped_tensor = tf.reshape(input_tensor, [batch_size, width * height, deep])
    gram = tf.matmul(reshaped_tensor, reshaped_tensor, transpose_a=True)
    num_elements = tf.cast(width * height * deep * channels, tf.float32)
    return gram / num_elements


def texture_loss(target_image, predicted_image):
    return tf.reduce_mean(tf.square(gram_matrix(target_image) - gram_matrix(predicted_image)))


def frequency_loss(target_image, predicted_image, dwt_layer):
    """L1 between the 3D DWT sub-bands; ``dwt_layer`` is a ``DWT3D(wavelet_name="bior1.3")``."""
    loss = tf.reduce_mean(tf.abs(dwt_layer(target_image) - dwt_layer(predicted_image)))
    return tf.cast(loss, tf.float32)


def cycle_loss(target_image, cycled_image):
    return tf.reduce_mean(tf.abs(target_image - cycled_image))


def apply_mask(image, seg):
    """Zero every voxel outside ``seg > 0``."""
    return tf.where(seg > 0, image, tf.zeros_like(image))


# ---------------------------------------------------------------- MPGAN (pix2pix, one direction)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return (real_loss + fake_loss) / 2


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def feature_extractor_loss(Fe, h4_real, h4_fake):
    """Perceptual loss: mean L1 between Models Genesis feature maps (computed in float16)."""
    feat_h4_real_list = Fe(h4_real, training=False)
    feat_h4_fake_list = Fe(h4_fake, training=False)

    loss = tf.constant(0.0, dtype=tf.float16)
    for feat_real, feat_fake in zip(feat_h4_real_list, feat_h4_fake_list):
        loss += tf.reduce_mean(tf.abs(feat_real - feat_fake))
    loss /= len(feat_h4_real_list)
    return tf.cast(loss, tf.float32)


def foreground_l1_loss(h4_real, h4_fake):
    """L1 over the voxels where the real image is positive (MPGAN_paper_16)."""
    return tf.reduce_mean(tf.abs(h4_real[h4_real > 0] - h4_fake[h4_real > 0]))


def voxel_loss(h4_real, h4_fake, seg_hunt4):
    """L1 over the voxels inside the segmentation (MPGAN_paper_16_seg_loss)."""
    return tf.reduce_mean(tf.abs(tf.boolean_mask((h4_real - h4_fake), seg_hunt4 > 0)))


# ---------------------------------------------------------------- segmentation loss (seg-loss variants)
def segmentation_dice_loss(batch_seg, batch_y_pred, means, stds, classs_weights, class_imbalance=None,
                           dtype=tf.float32):
    """Soft Dice between the true 3-tissue segmentation and the GMM tissue probabilities of the prediction.

    ``means``, ``stds`` and ``classs_weights`` are per-sample GMM parameters (shape ``(batch, 3)``)
    estimated on the real target image. ``class_imbalance`` is accepted for API compatibility but unused.
    """
    dice_loss = []
    for i in range(batch_seg.shape[0]):
        dice_loss.append(udl.gmm_dice_loss(tf.squeeze(batch_seg[i]),
                                           tf.squeeze(batch_y_pred[i]),
                                           means[i],
                                           stds[i],
                                           classs_weights[i],
                                           dtype=dtype))
    dice_loss = tf.convert_to_tensor(dice_loss)
    return tf.reduce_mean(dice_loss, axis=[0, 1])
