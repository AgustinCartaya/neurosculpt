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


def load_img(id, hunt):
    if hunt == 3:
        img, _ = nfc.load_nifti(cts.get_image_path_name(id, "0038", cts.INAME_REGISTERED))
        seg, _ = nfc.load_nifti(cts.get_image_path_name(id, "0038", cts.INAME_SEG_REGISTERED, nb_tissues=96))
        seg = ufs.merge_seg96_to_seg3(seg)
    else:
        img, _ = nfc.load_nifti(pil.get_registered_to_template_image_path_name(hunt, id, registration_to=pil.REGISTRATION_TO_MNI, absolute=True))
        seg, _ = nfc.load_nifti(pil.get_segmentation_path_name(hunt, id, 3, absolute=True))
    bmask_ncereb, _ = nfc.load_nifti(pil.get_brain_mask_path_name(hunt, id, pil.REGISTRATION_TO_MNI, no_cerebellum=True, absolute=True))
    img = util.skull_stripping_and_normalize(img, seg) 
    return img, seg, bmask_ncereb


def compute_whole_brain_patient_statistics(id):
    img_hunt3, seg_hunt3, bmask_ncereb_hunt3 = load_img(id, 3)
    img_hunt4, seg_hunt4, bmask_ncereb_hunt4 = load_img(id, 4)

    ncc = util.compute_normalized_cross_correlation(img_hunt3, img_hunt4)
    ssim = util.compute_ssim(img_hunt3, img_hunt4)
    psnr = util.compute_psnr(img_hunt3, img_hunt4)

    # compute bpf
    bpf_hunt3 = metrics_computer.compute_bpf(seg_hunt3, bmask_ncereb_hunt3)
    bpf_hunt4 = metrics_computer.compute_bpf(seg_hunt4, bmask_ncereb_hunt4)

    bpf_difference = bpf_hunt3 - bpf_hunt4
    bpf_division = bpf_hunt4 / bpf_hunt3

    return bpf_hunt3, bpf_hunt4, bpf_difference, bpf_division, ncc, ssim, psnr

def compute_patient_metrics_for_all():
    matching_ids = pil.get_matching_ids()

    columns_names = ['patient_id', 'bpf_hunt3', 'bpf_hunt4', 'bpf_dif', 'bpf_div',  'ncc', 'ssim', 'psnr']
    bpf_np = np.zeros((len(matching_ids), len(columns_names)))

    for i, id in enumerate(tqdm(matching_ids)):
        bpf_np[i, 0] = int(id)
        bpf_np[i, 1:] = compute_whole_brain_patient_statistics(id)
  
    df = pd.DataFrame(bpf_np, columns=columns_names)
    path_name = f'{current_path}/features/whole_brains_stats.csv'
    df.to_csv(path_name, index=False)

compute_patient_metrics_for_all()

