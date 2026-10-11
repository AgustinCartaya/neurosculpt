import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))

sys.path.append(base_path)

import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
from tqdm import tqdm
import pandas as pd
from IPython.display import display

import utils.util as util
import utils.nifti_functions as nfc
import utils.functions as fc
import pandas as pd

import utils.util_freesurfer_segmentation as ufs
import utils.util_brain_metrics as ubm


import project_images_loader as pil


import constants as cts


def save_stats(stats, ids, columns_names, table_name, prep_mni=True):
    ssim_df = pd.DataFrame(stats, index=ids, columns=columns_names)
    ssim_df.index.name = 'id'
    ssim_df.to_csv(cts.get_statistics_path_name(table_name, postprocessed=True, prep_mni=prep_mni), index=True, header=True)



def compute_statistics(ids, style_tyes_list, prep_mni=True):
    columns_names = [f"{style_type}" for style_type in style_tyes_list]
    print(columns_names)

    ssim_stats = np.zeros((len(ids), len(columns_names)))
    ncc_stats = np.zeros((len(ids), len(columns_names)))
    psnr_stats = np.zeros((len(ids), len(columns_names)))
    bpf_stats = np.zeros((len(ids), len(columns_names)))
    dices_stats = np.zeros((len(ids), len(columns_names), 3))

    for i, id in enumerate(tqdm(ids)):
        img_org, _ = nfc.load_nifti(pil.get_registered_to_template_image_path_name(4, id, pil.REGISTRATION_TO_MNI, absolute=True))
        seg_org, _ = nfc.load_nifti(pil.get_segmentation_path_name(4, id, nb_tissues=96, absolute=True))
        img_org = util.skull_stripping_and_normalize(img_org, seg_org)
        seg_org = ufs.remove_cerebellum(ufs.merge_seg96_to_seg3(seg_org), seg_org, deep_copy=False)

        for j, style_type in enumerate(style_tyes_list):
            
            img_res, _ = nfc.load_nifti(cts.get_result_image_path_name(id, "T1_POST", (style_type,), postprocessed=True, prep_mni=True))
            seg_res, _ = nfc.load_nifti(cts.get_result_image_path_name(id, "SEG_96_POST", (style_type,), postprocessed=True, prep_mni=True))
            bmask_res, _ = nfc.load_nifti(cts.get_result_image_path_name(id, "BMASK_NCRB_POST", (style_type,), postprocessed=True, prep_mni=True))

            img_res = util.skull_stripping_and_normalize(img_res, seg_res)
            seg_res = ufs.remove_cerebellum(ufs.merge_seg96_to_seg3(seg_res), seg_res, deep_copy=False)

            # # compute ssim
            # ssim_res = util.compute_ssim(img_org, img_res)
            # ssim_stats[i, j] = ssim_res

            #  # compute ncc
            ncc_res = util.compute_normalized_cross_correlation(img_org, img_res)
            ncc_stats[i, j] = ncc_res

            # compute psnr
            psnr_res = util.compute_psnr(img_org, img_res)
            psnr_stats[i, j] = psnr_res

            # # compute bpf
            # bpf_res = ubm.compute_bpf(seg_res, bmask_res)
            # bpf_stats[i, j] = bpf_res

            # # compute mean dice
            # dice_res = util.compute_multiple_dice_coefficients(seg_org, seg_res, return_dict=False)
            # dices_stats[i, j] = dice_res[1:]
                                 
            
    # save_stats(ssim_stats, ids, columns_names, "ssim" , prep_mni=prep_mni)
    save_stats(ncc_stats, ids, columns_names, "ncc" , prep_mni=prep_mni)
    save_stats(psnr_stats, ids, columns_names, "psnr" , prep_mni=prep_mni)
    # save_stats(bpf_stats, ids, columns_names, "bpf" , prep_mni=prep_mni)
    # save_stats(dices_stats[...,0], ids, columns_names, "dice_csf" , prep_mni=prep_mni)
    # save_stats(dices_stats[...,1], ids, columns_names, "dice_gm" , prep_mni=prep_mni)
    # save_stats(dices_stats[...,2], ids, columns_names, "dice_wm" , prep_mni=prep_mni)


val_images = ["08202"] + pil.get_matching_ids()[621:]
compute_statistics(val_images, ["BEST"], prep_mni=True)