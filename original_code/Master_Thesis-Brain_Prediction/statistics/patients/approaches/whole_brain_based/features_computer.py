import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../../../'))

sys.path.append(base_path)

import numpy as np
import pandas as pd
from tqdm import tqdm

import project_images_loader as pil

import utils.util as util
import utils.functions as fc
import utils.nifti_functions as nfc
import utils.util_freesurfer_segmentation as ufs
import test_code.dfield_approach_elastix.constants as cts
import utils.util_brain_metrics as metrics_computer

import test_code.dfield_approach_elastix.constants as reg_cts

# compute: bpf, ssim_hunt3, ssim_hunt4, ncc_hunt3, ncc_hunt4

INAME_RESULT_IMAGE = "{0}_0_{1}_{2}.nii.gz"


def load_org_img(id, hunt):
    img, _ = nfc.load_nifti(pil.get_registered_to_template_image_path_name(hunt, id, registration_to=pil.REGISTRATION_TO_MNI, absolute=True))
    seg, _ = nfc.load_nifti(pil.get_segmentation_path_name(hunt, id, 3, absolute=True))
    bmask_ncereb, _ = nfc.load_nifti(pil.get_brain_mask_path_name(hunt, id, pil.REGISTRATION_TO_MNI, no_cerebellum=True, absolute=True))
    img = util.skull_stripping_and_normalize(img, seg) 
    return img, bmask_ncereb

def load_result_img(id, approach, extra_info):
    extra_info_str = "_".join([str(info) for info in extra_info])
    path = f"{base_path}/approaches/{approach}/images/results_postprocessed/{id}"
    img_sufix = "_RAW" if approach == "MPGAN_paper_16_seg_loss" else "_POST"

    img, _ = nfc.load_nifti(f"{path}/{INAME_RESULT_IMAGE.format(id, 'T1'+img_sufix, extra_info_str)}")
    seg, _ = nfc.load_nifti(f"{path}/{INAME_RESULT_IMAGE.format(id, 'SEG_96'+img_sufix, extra_info_str)}")
    bmask_ncereb, _ = nfc.load_nifti(f"{path}/{INAME_RESULT_IMAGE.format(id, 'BMASK_NCRB'+img_sufix, extra_info_str)}")
    
    seg = ufs.merge_seg96_to_seg3(seg)
    img = util.skull_stripping_and_normalize(img, seg) 
    return img, seg, bmask_ncereb


def compute_whole_brain_patient_statistics(id, approach, extra_info):
    img_hunt3, bmask_ncereb_hunt3 = load_org_img(id, 3)
    img_hunt4, _  = load_org_img(id, 4)
    img_res, seg_res, _ = load_result_img(id, approach, extra_info)

    # compute ssim, ncc, psnr  
    ssim_hunt3 = util.compute_ssim(img_res, img_hunt3)
    ssim_hunt4 = util.compute_ssim(img_res, img_hunt4)
    ncc_hunt3 = util.compute_normalized_cross_correlation(img_res, img_hunt3)
    ncc_hunt4 = util.compute_normalized_cross_correlation(img_res, img_hunt4)
    psnr_hunt3 = util.compute_psnr(img_res, img_hunt3)
    psnr_hunt4 = util.compute_psnr(img_res, img_hunt4)

    # compute bpf
    bpf = metrics_computer.compute_bpf(seg_res, bmask_ncereb_hunt3)
    return bpf, ssim_hunt3, ssim_hunt4, ncc_hunt3, ncc_hunt4, psnr_hunt3, psnr_hunt4


def compute_patient_metrics_for_all(approach, extra_info):
    print(f"Computing metrics for approach {approach} and extra_info {extra_info}")
    if approach == "diffusion_model":
        val_images = pil.get_val_ids_2()
    else:
        val_images = ["08202"] + pil.get_matching_ids()[621:]

    columns_names = ['patient_id', 'bpf', 'ssim_hunt3', 'ssim_hunt4', 'ncc_hunt3', 'ncc_hunt4', 'psnr_hunt3', 'psnr_hunt4']
    bpf_np = np.zeros((len(val_images), len(columns_names)))

    for i, id in enumerate(tqdm(val_images)):
        bpf_np[i, 0] = int(id)
        bpf_np[i, 1:] = compute_whole_brain_patient_statistics(id, approach, extra_info)
        
        
    df = pd.DataFrame(bpf_np, columns=columns_names)
    path_name = f'{current_path}/features/{approach}/whole_brains_stats.csv'
    df.to_csv(path_name, index=False)

# compute_patient_metrics_for_all("dfield_similar_patches", [96, 10])
# compute_patient_metrics_for_all("MGAN_paper_16", [46])
# compute_patient_metrics_for_all("MPGAN_paper_16_seg_loss", [52])
# compute_patient_metrics_for_all("StarGAN_V2_sagital", ["BEST"])
# compute_patient_metrics_for_all("diffusion_model", ["RES"])
compute_patient_metrics_for_all("MPGAN_paper_16", [28])


