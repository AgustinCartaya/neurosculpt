import os
import argparse
import sys

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(base_path)


import numpy as np
import tensorflow as tf
from generator_IN import MetamorphicGenerator
from tqdm import tqdm

import project_images_loader as pil
import utils.util_deep_learning as udl
import constants as cts
import utils.functions as fc
import utils.nifti_functions as nfc
import utils.patches_controller as patches_controller

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')


def load_generator(checkpoint_index):
    G_h4 = MetamorphicGenerator()
    checkpoint_dir = f"{base_path}/test_code/MGAN/MGAN_paper_16_seg_loss/models/training_checkpoints/"
    checkpoint = tf.train.Checkpoint(G_h4=G_h4)
    checkpoint.restore(f"{checkpoint_dir}/ckpt-{checkpoint_index}")
    return G_h4


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

def predict_image(id, G, patch_size=(64,64,64), patch_step=(32,32,32)):
    img_hunt3, affine = pil.load_preprocessed_image(3, id, absolute=True)
    img_gen = generate_image_step(img_hunt3, G, patch_size=patch_size, patch_step=patch_step)
    return img_gen, affine


def predict_all(val_ids):
    # for checkpoint in range(66, 69, 2):
    for checkpoint in range(1, 66, 8):
        G = load_generator(checkpoint)
        for id in val_ids:
            print(f"predic {id} checkpoint {checkpoint}")
            result_path_name = cts.get_result_image_path_name(id, "T1", [checkpoint])
            if fc.verify_folder(os.path.dirname(result_path_name), create=True):
                img_gen, affine = predict_image(id, G)
                nfc.save_nifti(img_gen, affine, result_path_name)
            break
val_images = ["08202"] + pil.get_matching_ids()[621:]
predict_all(val_images)