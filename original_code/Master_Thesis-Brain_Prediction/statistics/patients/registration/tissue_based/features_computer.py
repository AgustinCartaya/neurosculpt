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
import utils.nifti_functions as nfc
import test_code.dfield_approach_elastix.constants as cts
import metrics_computer as metrics_computer
import utils.util_freesurfer_segmentation as ufs


np.set_printoptions(precision=4)

JACOBIAN = 'jac'
DIVERGENCE = 'div'
MAGNITUDE = 'mag'

def get_colums_names(att):
    return [f'{att}_min', f'{att}_max', f'{att}_mean', f'{att}_std']



def load_segmentation(id, hunt, nb_tissues, remove_cerebellum=False):
    if hunt == 3:
        seg, _ = nfc.load_nifti(cts.get_image_path_name(id, "0038", cts.INAME_SEG_REGISTERED, nb_tissues=96))
    else:
        seg, _ = nfc.load_nifti(pil.get_segmentation_path_name(hunt, id, nb_tissues=96, absolute=True))
    
    if remove_cerebellum:
        seg = ufs.remove_cerebellum(seg, seg, deep_copy=False)
    
    if nb_tissues == 3:
        seg = ufs.merge_seg96_to_seg3(seg)
    elif nb_tissues == 36:
        seg = ufs.merge_seg96_to_seg36(seg)
    return seg


def compute_patient_metrics(id, nb_tissues):
    seg_base = load_segmentation(id, 3, nb_tissues=nb_tissues, remove_cerebellum=(nb_tissues==3))
    seg_fup = load_segmentation(id, 4, nb_tissues=nb_tissues, remove_cerebellum=(nb_tissues==3))
    
    seg_indexs = np.unique(np.concatenate([seg_base, seg_fup]).astype(np.uint))
    seg_indexs = np.sort(seg_indexs)

    tissue_metrics = {}
    for tissue_id in np.unique(seg_indexs):
        mask_base = seg_base == tissue_id
        mask_fup = seg_fup == tissue_id

        _dice = util.compute_dice_coefficient(mask_base, mask_fup)
        _vol_diff = np.sum(mask_fup) - np.sum(mask_base)
        _vol_fraction = np.sum(mask_fup) / np.sum(mask_base)
        _vol_diff_fraction = _vol_diff / np.sum(mask_base)

        tissue_metrics[tissue_id] = (_dice, _vol_diff, _vol_fraction, _vol_diff_fraction)

    columns_names = ['dice', 'vol_diff', 'vol_fraction', 'vol_diff_fraction'] 
    tissue_metrics_df = pd.DataFrame.from_dict(tissue_metrics, orient='index', columns=columns_names)
    tissue_metrics_df.index.name = 'tissue_id'

    # save results
    path_name = f'{current_path}/features/tissues_{nb_tissues}/{id}.csv'
    tissue_metrics_df.to_csv(path_name)

def compute_patient_metrics_for_all():
    # matching_ids = pil.get_matching_ids()
    matching_ids = ["08202"] + pil.get_matching_ids()[621:]

    for nb_tissues in [96]:
        print(f'Computing metrics for {nb_tissues} tissues')
        for matching_id in tqdm(matching_ids):
            compute_patient_metrics(matching_id, nb_tissues=nb_tissues)
    
compute_patient_metrics_for_all()