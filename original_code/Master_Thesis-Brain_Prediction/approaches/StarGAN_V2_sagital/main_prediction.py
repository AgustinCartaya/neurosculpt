"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import argparse
import sys

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(base_path)


from munch import Munch
from torch.backends import cudnn
import torch
import numpy as np
import project_images_loader as pil
from data_loaders.data_loaders_2D.data_loader_stargan_2D_images_predict import DataLoaderImagesStarGAN2DPredict
from data_loaders.data_loaders_2D.data_loader_stargan_2D_input_fetcher_predict import InputFetcherPredict
from core.solver import Solver
import utils.functions as fc
from tqdm import tqdm

import utils.nifti_functions as nfc
import test_code.creator_2D_images.constants as cts_prep_mni
from core.checkpoint import CheckpointIO

from os.path import join as ospj
from core.model import build_model

import constants as cts
features_loader_tissue_based = fc.load_module_from_path('features_loader_tissue_based', os.path.join(base_path, 'statistics/patients/inter_patient/tissue_based/features_loader.py'))
features_loader_bpf = fc.load_module_from_path('features_loader_bpf', os.path.join(base_path, 'statistics/patients/cross_sectional/whole_brain/features_loader.py'))

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)
cudnn.benchmark = True
torch.manual_seed(777)



def load_networks(args, step):
    checkpoint_dir = f'{current_path}/expr/checkpoints'
    _, nets_ema = build_model(args)
    ckptio = CheckpointIO(ospj(checkpoint_dir, '{:06d}_nets_ema.ckpt'.format(step)), data_parallel=True, **nets_ema)
    ckptio.load(step)
    return nets_ema


def create_fetcher(img_size, img_src, img_sty, bmask_src, bmask_sty, hunt_src, hunt_sty):


    data_train = DataLoaderImagesStarGAN2DPredict(img_size=(img_size,img_size), 
                                        img_src=img_src,
                                        img_sty=img_sty, 
                                        mask_src=bmask_src,
                                        mask_sty=bmask_sty, 
                                        hunt_src=hunt_src, 
                                        hunt_sty=hunt_sty, 
                                        axis=0,
                                        channels_first=True)

    data_fetcher = InputFetcherPredict(data_train)
    return data_fetcher

@torch.no_grad()
def predict3D(style_encoder, generator, fetcher_pred):
    def denormalize(x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    
    predicted_layers = []
    for inputs in tqdm(fetcher_pred):
        x_real, y_org, mask_real = inputs.x_src, inputs.y_src, inputs.mask
        x_ref, y_ref, mask_ref = inputs.x_ref,  inputs.y_ref, inputs.mask_ref

        s_ref = style_encoder(x_ref, y_ref)
        x_fake = generator(x_real, s_ref, masks=mask_real)
        x_fake = denormalize(x_fake)
        _prediction = np.squeeze(x_fake.cpu().numpy())
        predicted_layers.append(_prediction)
    return predicted_layers


def load_images(id_src, id_sty, hunt_src, hunt_sty, prep_mni):
    if prep_mni:
        img_src, affine = nfc.load_nifti(cts_prep_mni.get_registered_image_path_name(hunt_src+3, id_src))
        bmask_src = nfc.load_nifti(cts_prep_mni.get_registered_bmask_path_name(hunt_src+3, id_src))[0]

        img_sty = nfc.load_nifti(cts_prep_mni.get_registered_image_path_name(hunt_sty+3, id_sty))[0]
        bmask_sty = nfc.load_nifti(cts_prep_mni.get_registered_bmask_path_name(hunt_sty+3, id_sty))[0]
    else:
        img_src, affine = nfc.load_nifti(pil.get_preprocessed_image_path_name(hunt_src+3, id_src, absolute=True))
        bmask_src = nfc.load_nifti(pil.get_brain_mask_path_name(hunt_src+3, id_src, pil.REGISTRATION_TO_MNI, no_cerebellum=False, absolute=True))[0]

        img_sty = nfc.load_nifti(pil.get_preprocessed_image_path_name(hunt_sty+3, id_sty, absolute=True))[0]
        bmask_sty = nfc.load_nifti(pil.get_brain_mask_path_name(hunt_sty+3, id_sty, pil.REGISTRATION_TO_MNI, no_cerebellum=False, absolute=True))[0]

    return img_src, img_sty, bmask_src, bmask_sty, affine


def load_and_predict(id_src, id_sty, step, prep_mni, hunt_sty=None):
    hunt_src = DataLoaderImagesStarGAN2DPredict.HUNT3 
    if hunt_sty is None:
        hunt_sty =  DataLoaderImagesStarGAN2DPredict.HUNT4

    img_src, img_sty, bmask_src, bmask_sty, affine = load_images(id_src, id_sty, hunt_src, hunt_sty, prep_mni=prep_mni)
    
    args = Munch(img_size=256, style_dim=64, w_hpf=1, n_channels=1, latent_dim=16, num_domains=2)
    
    nets_ema = load_networks(args, step=step)
    fetcher_pred = create_fetcher(args.img_size, img_src, img_sty, bmask_src, bmask_sty, hunt_src, hunt_sty)
    predicted_layers = predict3D(nets_ema.style_encoder, nets_ema.generator, fetcher_pred)
    predicted_layers = [fc.remove_padding(layer, img_src.shape[1:]) for layer in predicted_layers]

    img_pred = np.stack(predicted_layers, axis=0)
    return img_pred, affine


def save_predictions_using_gt(ids, step, prep_mni):
    for id in ids:
        print(f"predicting {id} using {id}")
        result_path_name = cts.get_result_image_path_name(id, "T1", ["GT"], prep_mni=prep_mni)

        # verify if already exists
        if fc.verify_file(result_path_name):
            continue
        
        # predict
        if fc.verify_folder(os.path.dirname(result_path_name), create=True):
            img_pred, affine = load_and_predict(id, id, step, prep_mni)
            nfc.save_nifti(img_pred, affine, result_path_name)


def save_predictions_using_most_similar_image(ids, step, prep_mni):
    matching_ids = pil.get_matching_ids()
    dice_mean_table_reder = features_loader_tissue_based.FeaturesLoader(3, matching_ids)

    for id in ids:
        # find most similar image
        df_patient_sim_table = dice_mean_table_reder.get_sorted_similarity_table(id, "dice_mean", 1, exclude_ids= ids + pil.EXCLUDED_PATIENTS)
        id_sty = df_patient_sim_table.iloc[0]['id']

        print(f"predicting {id} using {id_sty}")
        result_path_name = cts.get_result_image_path_name(id, "T1", ["BEST"], prep_mni=prep_mni)
        
        # verify if already exists
        if fc.verify_file(result_path_name):
            continue

        # predict
        if fc.verify_folder(os.path.dirname(result_path_name), create=True):
            img_pred, affine = load_and_predict(id, id_sty, step, prep_mni)
            nfc.save_nifti(img_pred, affine, result_path_name)


def save_predictions_using_differents_bpf(ids, step, prep_mni, bpf_type="HIGH"):
    # find img with required bpf
    bpf_reader = features_loader_bpf.FeaturesLoader()
    bpf_table = bpf_reader.get_df(columns=["bpf_hunt3"], remove_columns=False, return_patient_id=True, remove_patients=ids + pil.EXCLUDED_PATIENTS)

    if bpf_type == "HIGH":
        bpf_index = bpf_table['bpf_hunt3'].idxmax()
    elif bpf_type == "LOW":
        bpf_index = bpf_table['bpf_hunt3'].idxmin()
    id_sty = int(bpf_table.at[bpf_index, 'patient_id'])
    id_sty = f"{id_sty:05d}"
 
    for id in ids:
        print(f"predicting {id} using {id_sty} BPF: {bpf_type}")
        result_path_name = cts.get_result_image_path_name(id, "T1", [id_sty, bpf_type], prep_mni=prep_mni)
        
        # verify if already exists
        if fc.verify_file(result_path_name):
            continue

        # predict
        if fc.verify_folder(os.path.dirname(result_path_name), create=True):
            img_pred, affine = load_and_predict(id, id_sty, step, prep_mni)
            nfc.save_nifti(img_pred, affine, result_path_name)
        break


def save_predictions_using_self(ids, step, prep_mni, hunt=DataLoaderImagesStarGAN2DPredict.HUNT4):
    # find img with required bpf
    for id in ids:
        print(f"predicting {id} using {id} SELF")
        result_path_name = cts.get_result_image_path_name(id, "T1", [id, "SELF", f"STYLE_HUNT{hunt+3}"], prep_mni=prep_mni)
        # verify if already exists
        if fc.verify_file(result_path_name):
            continue
        # predict
        if fc.verify_folder(os.path.dirname(result_path_name), create=True):
            img_pred, affine = load_and_predict(id, id, step, prep_mni, hunt_sty=hunt)
            nfc.save_nifti(img_pred, affine, result_path_name)

def save_predictions_custom_id(ids, step, prep_mni, id_sty):
    # find img with required bpf
    for id in ids:
        print(f"predicting {id} using CUSTOM {id_sty}")
        result_path_name = cts.get_result_image_path_name(id, "T1", [id_sty, f"CUSTOM"], prep_mni=prep_mni)
        # verify if already exists
        if fc.verify_file(result_path_name):
            continue
        # predict
        if fc.verify_folder(os.path.dirname(result_path_name), create=True):
            img_pred, affine = load_and_predict(id, id, step, prep_mni)
            nfc.save_nifti(img_pred, affine, result_path_name)
        break

val_images = ["08202"] + pil.get_matching_ids()[621:]
# # save_predictions_using_gt(val_images, step=65200, prep_mni=True) # done
# save_predictions_using_most_similar_image(val_images, step=65200, prep_mni=True) # done
# save_predictions_using_differents_bpf(val_images, step=65200, prep_mni=True, bpf_type="HIGH") # not working
# save_predictions_using_differents_bpf(val_images, step=65200, prep_mni=True, bpf_type="LOW") # not working
# save_predictions_using_self(val_images, step=65200, prep_mni=True, hunt=DataLoaderImagesStarGAN2DPredict.HUNT3) # not working
save_predictions_custom_id(val_images, step=65200, prep_mni=True, id_sty="00039")

# save_predictions_using_gt(val_images, step=65200, prep_mni=False) # not important
# save_predictions_using_most_similar_image(val_images, step=65200, prep_mni=True) # not important