import os
import argparse
import sys

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(base_path)


import numpy as np
import tensorflow as tf
import generator as Generator

import project_images_loader as pil
import utils.util_deep_learning as udl
import constants as cts
import utils.functions as fc
import utils.nifti_functions as nfc

tf.keras.backend.set_image_data_format("channels_first")
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')


def load_generator(checkpoint_index):
    input_shape = (192,224,192)
    G = Generator.create_generator_model(input_shape)
    checkpoint_dir = f"{base_path}/test_code/MPGAN/MPGAN_paper_16_seg_loss/models/training_checkpoints"
    checkpoint = tf.train.Checkpoint(G=G)
    checkpoint.restore(f"{checkpoint_dir}/ckpt-{checkpoint_index}")
    return G

def predict_image(id, G):
    img_hunt3, affine = pil.load_preprocessed_image(3, id, absolute=True)
    img_gen = G(udl.prepare_img_for_gan(img_hunt3, axis_channels=1), training=False)
    img_gen = udl.recover_img_from_gan(img_gen, axis_channels=1)
    return img_gen, affine


def predict_all(val_ids):
    for checkpoint in range(1, 46, 4):
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