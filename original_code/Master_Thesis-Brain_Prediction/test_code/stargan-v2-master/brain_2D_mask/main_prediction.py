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
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../../'))
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

import utils.nifti_functions as nfc
import test_code.creator_2D_images.constants as cts
from core.checkpoint import CheckpointIO

from os.path import join as ospj
from core.model import build_model


# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)
cudnn.benchmark = True
torch.manual_seed(777)



def load_networks(args, step=57600):
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
    for i, inputs in enumerate(fetcher_pred):
        x_real, y_org, mask_real = inputs.x_src, inputs.y_src, inputs.mask
        x_ref, y_ref, mask_ref = inputs.x_ref,  inputs.y_ref, inputs.mask_ref

        s_ref = style_encoder(x_ref, y_ref)
        x_fake = generator(x_real, s_ref, masks=mask_real)
        x_fake = denormalize(x_fake)
        _prediction = np.squeeze(x_fake.cpu().numpy())
        predicted_layers.append(_prediction)
        print(f"Layer {i} predicted")
    return predicted_layers



    
def tets_predict():
    import matplotlib.pyplot as plt
    matching_ids = pil.get_matching_ids() #"08202" = 366
    id_src = matching_ids[700]
    id_sty = matching_ids[300]
    hunt_src = DataLoaderImagesStarGAN2DPredict.HUNT3
    hunt_sty =  DataLoaderImagesStarGAN2DPredict.HUNT4

    # img_src = nfc.load_nifti(cts.get_registered_image_path_name(hunt_src+3, id_src))[0]
    # bmask_src = nfc.load_nifti(cts.get_registered_bmask_path_name(hunt_src+3, id_src))[0]

    # img_sty = nfc.load_nifti(cts.get_registered_image_path_name(hunt_sty+3, id_sty))[0]
    # bmask_sty = nfc.load_nifti(cts.get_registered_bmask_path_name(hunt_sty+3, id_sty))[0]
    
    img_src = nfc.load_nifti(pil.get_preprocessed_image_path_name(hunt_src+3, id_src, absolute=True))[0]
    img_src_real = nfc.load_nifti(pil.get_preprocessed_image_path_name((1-hunt_src)+3, id_src, absolute=True))[0]
    bmask_src = nfc.load_nifti(pil.get_brain_mask_path_name(hunt_src+3, id_src, pil.REGISTRATION_TO_MNI, no_cerebellum=False, absolute=True))[0]

    img_sty = nfc.load_nifti(pil.get_preprocessed_image_path_name(hunt_sty+3, id_sty, absolute=True))[0]
    bmask_sty = nfc.load_nifti(pil.get_brain_mask_path_name(hunt_sty+3, id_sty, pil.REGISTRATION_TO_MNI, no_cerebellum=False, absolute=True))[0]


    args = Munch(img_size=256, style_dim=64, w_hpf=1, n_channels=1, latent_dim=16, num_domains=2)
    
    nets_ema = load_networks(args, step=57600)
    fetcher_pred = create_fetcher(args.img_size, img_src, img_sty, bmask_src, bmask_sty, hunt_src, hunt_sty)
    predicted_layers = predict3D(nets_ema.style_encoder, nets_ema.generator, fetcher_pred)
    predicted_layers = [fc.remove_padding(layer, img_src.shape[1:]) for layer in predicted_layers]

    img_pred = np.stack(predicted_layers, axis=0)
    print(img_pred.shape)
    img_pred *= bmask_src

    current_plane = 116
    fc.imgshow_3D("img_src", img_src, show=False, current_plane=current_plane)
    fc.imgshow_3D("img_src_real", img_src_real, show=False, current_plane=current_plane)
    fc.imgshow_3D("img_sty", img_sty, show=False, current_plane=current_plane)
    fc.imgshow_3D("img_pred", img_pred, show=False, current_plane=current_plane)
    plt.show()

tets_predict()