"""Checkpoint loading and whole-image prediction for the MGAN (patch-based) and MPGAN (whole-volume) generators.

Call :func:`brainpred.methods.gan.variants.configure_tensorflow` before using these functions.
"""
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from brainpred import paths
from brainpred.methods.gan import variants as V
from brainpred.utils import deep_learning as udl
from brainpred.utils import patches as patches_controller


# ---------------------------------------------------------------- checkpoint loading
def load_generator(variant, checkpoint_index, checkpoint_dir=None):
    """Restore the HUNT3 -> HUNT4 generator of ``variant`` from ``<checkpoint_dir>/ckpt-<checkpoint_index>``."""
    checkpoint_dir = checkpoint_dir or V.get_checkpoint_dir(variant)
    checkpoint_path = f"{checkpoint_dir}/ckpt-{checkpoint_index}"
    print(f"Loading {checkpoint_path}")

    if variant.family == V.FAMILY_MGAN:
        from brainpred.methods.gan.mgan.generator import MetamorphicGenerator
        G = MetamorphicGenerator(norm=variant.norm)
        # The training checkpoint also holds G_h3, both discriminators and the optimizers.
        tf.train.Checkpoint(G_h4=G).restore(checkpoint_path).expect_partial()
    else:
        from brainpred.methods.gan.mpgan.generator import create_generator_model
        G = create_generator_model(V.MPGAN_INPUT_SHAPE, norm=variant.norm)
        tf.train.Checkpoint(G=G).restore(checkpoint_path).expect_partial()
    return G


# ---------------------------------------------------------------- generation
def generate_image_patches(img_org, G, patch_size=V.MGAN_PATCH_SIZE, patch_step=V.MGAN_PATCH_STEP, batch_size=20):
    """MGAN: predict overlapping patches (finest output scale) and merge them by weighted voting."""
    img_patches = patches_controller.extract_patches(img_org, patch_shape=patch_size,
                                                     extraction_step=patch_step).astype(np.float32)
    gen_patches = np.zeros_like(img_patches)

    for i in tqdm(range(0, len(img_patches), batch_size)):
        end = min(i + batch_size, len(img_patches))
        gen_result = G(np.expand_dims(img_patches[i:end], axis=-1), training=False)
        gen_patches[i:end] = gen_result[-1][..., 0]

    return patches_controller.perform_voting(gen_patches,
                                             patch_shape=patch_size,
                                             expected_shape=img_org.shape,
                                             extraction_step=patch_step,
                                             window_sep=patch_step)


def generate_image_full(img_org, G, img_size=V.MPGAN_INPUT_SHAPE, img_offset=V.MPGAN_IMG_OFFSET):
    """MPGAN: crop to ``img_size``, predict in one pass, paste back into a (193, 229, 193) volume."""
    img_gen = G(udl.prepare_img_for_gan(img_org, axis_channels=1, img_size=img_size, img_offset=img_offset),
                training=False)
    return udl.recover_img_from_gan(img_gen, axis_channels=1, img_offset=img_offset)


def generate_image(variant, img_org, G):
    if variant.family == V.FAMILY_MGAN:
        return generate_image_patches(img_org, G)
    return generate_image_full(img_org, G)


def predict_image(variant, id, G):
    """Predict the HUNT4 scan of patient ``id`` from its preprocessed HUNT3 scan; returns ``(img, affine)``."""
    img_hunt3, affine = paths.load_preprocessed_image(3, id)
    return generate_image(variant, img_hunt3, G), affine
