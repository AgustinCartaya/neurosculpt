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
import utils.util_freesurfer_segmentation as ufs
import utils.util_brain_metrics as ubm
import pandas as pd


import project_images_loader as pil


import constants as cts

def save_stats(stats, ids, columns_names, table_name):
    ssim_df = pd.DataFrame(stats, index=ids, columns=columns_names)
    ssim_df.index.name = 'id'
    ssim_df.to_csv(cts.get_statistics_path_name(table_name), index=True, header=True)


def compute_statistics(ids, nb_images_list, reg_type_list):
    columns_names = [f"{reg_type.upper()}_{nb_images}" for reg_type in reg_type_list for nb_images in nb_images_list]
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

        index = 0
        for j, reg_type in enumerate(reg_type_list):
            for k, nb_images in enumerate(nb_images_list):
                img_res, _ = nfc.load_nifti(cts.get_result_image_path_name(id, "T1", (reg_type, nb_images)))
                seg_res, _ = nfc.load_nifti(cts.get_result_image_path_name(id, "SEG_96", (reg_type, nb_images)))
                bmask_res, _ = nfc.load_nifti(cts.get_result_image_path_name(id, "BMASK_NCRB", (reg_type, nb_images)))
                # merge segmentation to 3 tissues
                img_res = util.skull_stripping_and_normalize(img_res, seg_res)
                seg_res = ufs.remove_cerebellum(ufs.merge_seg96_to_seg3(seg_res), seg_res, deep_copy=False)
                # nfc.save_nifti(seg_res, seg_aff, pil.get_temp_folder()+f"/{id}_seg_res_{j}_{k}.nii.gz")

                # # compute ssim
                # ssim_res = util.compute_ssim(img_org, img_res)
                # ssim_stats[i, index] = ssim_res
                
                # compute ncc
                ncc_res = util.compute_normalized_cross_correlation(img_org, img_res)
                ncc_stats[i, index] = ncc_res

                # compute psnr
                psnr_res = util.compute_psnr(img_org, img_res)
                psnr_stats[i, index] = psnr_res

                # # compute bpf
                # bpf_res = ubm.compute_bpf(seg_res, bmask_res)
                # bpf_stats[i, index] = bpf_res

                # # compute mean dice
                # dice_res = util.compute_multiple_dice_coefficients(seg_org, seg_res, return_dict=False)
                # dices_stats[i, index] = dice_res[1:]

                index += 1
         
        
    # save_stats(ssim_stats, ids, columns_names, "ssim")
    save_stats(ncc_stats, ids, columns_names, "ncc")
    save_stats(psnr_stats, ids, columns_names, "psnr")
    # save_stats(bpf_stats, ids, columns_names, "bpf")
    # save_stats(dices_stats[...,0], ids, columns_names, "dice_csf")
    # save_stats(dices_stats[...,1], ids, columns_names, "dice_gm")
    # save_stats(dices_stats[...,2], ids, columns_names, "dice_wm")

val_images = ["08202"] + pil.get_matching_ids()[621:]
compute_statistics(val_images, [5], [cts.PARAMETERS_FILE_AFFINE, cts.PARAMETERS_FILE_BSPLINE_FAST])