"""Longitudinal whole-brain features: BPF at both visits, its change, and HUNT3-vs-HUNT4 image similarity."""
import numpy as np
import pandas as pd
from tqdm import tqdm

from brainpred import paths, splits
from brainpred.features.common import WHOLE_BRAIN_TABLE_NAME, WholeBrainTableLoader, get_features_folder, save_table
from brainpred.utils import brain_metrics, intensity, io, metrics

COLUMNS = ['patient_id', 'bpf_hunt3', 'bpf_hunt4', 'bpf_dif', 'bpf_div', 'ncc', 'ssim', 'psnr']


def get_table_path_name():
    return f'{get_features_folder("longitudinal", "whole_brain_based")}/{WHOLE_BRAIN_TABLE_NAME}.csv'


def load_img(id, hunt):
    img, _ = io.load_nifti(paths.get_registered_to_template_image_path_name(hunt, id, registration_to=paths.REGISTRATION_TO_MNI))
    seg, _ = io.load_nifti(paths.get_segmentation_path_name(hunt, id, 3))
    bmask_ncereb, _ = io.load_nifti(paths.get_brain_mask_path_name(hunt, id, paths.REGISTRATION_TO_MNI, no_cerebellum=True))
    img = intensity.skull_stripping_and_normalize(img, seg)
    return img, seg, bmask_ncereb


def compute_pair_statistics(img_hunt3, seg_hunt3, bmask_ncereb_hunt3, img_hunt4, seg_hunt4, bmask_ncereb_hunt4):
    """BPF of both visits, their difference / ratio and the NCC, SSIM, PSNR between both images."""
    ncc = metrics.compute_normalized_cross_correlation(img_hunt3, img_hunt4)
    ssim = metrics.compute_ssim(img_hunt3, img_hunt4)
    psnr = metrics.compute_psnr(img_hunt3, img_hunt4)

    bpf_hunt3 = brain_metrics.compute_bpf(seg_hunt3, bmask_ncereb_hunt3)
    bpf_hunt4 = brain_metrics.compute_bpf(seg_hunt4, bmask_ncereb_hunt4)

    bpf_difference = bpf_hunt3 - bpf_hunt4
    bpf_division = bpf_hunt4 / bpf_hunt3
    return bpf_hunt3, bpf_hunt4, bpf_difference, bpf_division, ncc, ssim, psnr


def compute_table(load_img_fn, path_name):
    """Compute :data:`COLUMNS` for every matching patient with ``load_img_fn(id, hunt)``."""
    matching_ids = splits.get_matching_ids()
    bpf_np = np.zeros((len(matching_ids), len(COLUMNS)))
    for i, id in enumerate(tqdm(matching_ids)):
        bpf_np[i, 0] = int(id)
        bpf_np[i, 1:] = compute_pair_statistics(*load_img_fn(id, 3), *load_img_fn(id, 4))
    save_table(pd.DataFrame(bpf_np, columns=COLUMNS), path_name)


def compute_patient_metrics_for_all():
    compute_table(load_img, get_table_path_name())


class FeaturesLoader(WholeBrainTableLoader):
    def __init__(self):
        super().__init__(get_table_path_name())
