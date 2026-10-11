"""Whole-brain features of a method's postprocessed predictions: BPF and SSIM / NCC / PSNR against the real HUNT3 and HUNT4."""
import numpy as np
import pandas as pd
from tqdm import tqdm

from brainpred import paths
from brainpred.features.approaches.common import get_approach_ids, load_result_image
from brainpred.features.common import WHOLE_BRAIN_TABLE_NAME, WholeBrainTableLoader, get_features_folder, save_table
from brainpred.utils import brain_metrics, intensity, io, metrics
from brainpred.utils import segmentation as ufs

COLUMNS = ['patient_id', 'bpf', 'ssim_hunt3', 'ssim_hunt4', 'ncc_hunt3', 'ncc_hunt4', 'psnr_hunt3', 'psnr_hunt4']


def get_table_path_name(approach):
    return f'{get_features_folder("approaches", "whole_brain_based", approach)}/{WHOLE_BRAIN_TABLE_NAME}.csv'


def load_org_img(id, hunt):
    img, _ = io.load_nifti(paths.get_registered_to_template_image_path_name(hunt, id, registration_to=paths.REGISTRATION_TO_MNI))
    seg, _ = io.load_nifti(paths.get_segmentation_path_name(hunt, id, 3))
    bmask_ncereb, _ = io.load_nifti(paths.get_brain_mask_path_name(hunt, id, paths.REGISTRATION_TO_MNI, no_cerebellum=True))
    img = intensity.skull_stripping_and_normalize(img, seg)
    return img, bmask_ncereb


def load_result_img(id, approach, extra_info):
    img = load_result_image(id, approach, 'T1', extra_info)
    seg = load_result_image(id, approach, 'SEG_96', extra_info)
    bmask_ncereb = load_result_image(id, approach, 'BMASK_NCRB', extra_info)

    seg = ufs.merge_seg96_to_seg3(seg)
    img = intensity.skull_stripping_and_normalize(img, seg)
    return img, seg, bmask_ncereb


def compute_whole_brain_patient_statistics(id, approach, extra_info):
    img_hunt3, bmask_ncereb_hunt3 = load_org_img(id, 3)
    img_hunt4, _ = load_org_img(id, 4)
    img_res, seg_res, _ = load_result_img(id, approach, extra_info)

    ssim_hunt3 = metrics.compute_ssim(img_res, img_hunt3)
    ssim_hunt4 = metrics.compute_ssim(img_res, img_hunt4)
    ncc_hunt3 = metrics.compute_normalized_cross_correlation(img_res, img_hunt3)
    ncc_hunt4 = metrics.compute_normalized_cross_correlation(img_res, img_hunt4)
    psnr_hunt3 = metrics.compute_psnr(img_res, img_hunt3)
    psnr_hunt4 = metrics.compute_psnr(img_res, img_hunt4)

    # BPF of the prediction with the HUNT3 brain mask (without cerebellum)
    bpf = brain_metrics.compute_bpf(seg_res, bmask_ncereb_hunt3)
    return bpf, ssim_hunt3, ssim_hunt4, ncc_hunt3, ncc_hunt4, psnr_hunt3, psnr_hunt4


def compute_patient_metrics_for_all(approach, extra_info):
    print(f"Computing metrics for approach {approach} and extra_info {extra_info}")
    val_images = get_approach_ids(approach)

    bpf_np = np.zeros((len(val_images), len(COLUMNS)))
    for i, id in enumerate(tqdm(val_images)):
        bpf_np[i, 0] = int(id)
        bpf_np[i, 1:] = compute_whole_brain_patient_statistics(id, approach, extra_info)

    save_table(pd.DataFrame(bpf_np, columns=COLUMNS), get_table_path_name(approach))


class FeaturesLoader(WholeBrainTableLoader):
    def __init__(self, approach):
        super().__init__(get_table_path_name(approach))
