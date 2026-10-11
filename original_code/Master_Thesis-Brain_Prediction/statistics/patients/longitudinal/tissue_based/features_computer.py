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
import utils.util_dfield_metrics as metrics_computer
import utils.util_freesurfer_segmentation as ufs
import utils.functions as fc


np.set_printoptions(precision=4)

JACOBIAN = 'jac'
DIVERGENCE = 'div'
MAGNITUDE = 'mag'

def get_colums_names(att):
    return [f'{att}_min', f'{att}_max', f'{att}_mean', f'{att}_std']


def load_segmentation(id, hunt, nb_tissues):
    seg, _ = nfc.load_nifti(pil.get_segmentation_path_name(hunt, id, nb_tissues=96, absolute=True))
    if nb_tissues == 3:
        seg = ufs.remove_cerebellum(ufs.merge_seg96_to_seg3(seg), seg, deep_copy=False)
    elif nb_tissues == 36:
        seg = ufs.merge_seg96_to_seg36(seg)
    return seg


def compute_patient_metrics(id, nb_tissues):
    seg_base = load_segmentation(id, 3, nb_tissues=nb_tissues)
    seg_fup = load_segmentation(id, 4, nb_tissues=nb_tissues)
    
    dfield, dfield_sitk = nfc.load_image_sitk(cts.get_image_path_name(id, cts.PFILE_0038, cts.INAME_DFIELD))

    tissue_jac_dict = metrics_computer.compute_dfield_jacobian(dfield_sitk)
    tissue_div_dict = metrics_computer.compute_dfield_divergence(dfield)
    tissue_mag_dict = metrics_computer.compute_dfield_magnitude(dfield)

    seg_indexs = np.unique(seg_fup.astype(np.uint))
    seg_indexs = np.sort(seg_indexs)

    tissue_metrics = {}
    for tissue_id in np.unique(seg_indexs):
        mask_base = seg_base == tissue_id
        mask_fup = seg_fup == tissue_id

        jac_i = tissue_jac_dict[mask_base]
        div_i = tissue_div_dict[mask_base]
        mag_i = tissue_mag_dict[mask_base]

        _jac_metrics = util.get_basic_metrics(jac_i)
        _div_metrics = util.get_basic_metrics(div_i)
        _mag_metrics = util.get_basic_metrics(mag_i)

        _dice = util.compute_dice_coefficient(mask_base, mask_fup)
        _vol_diff = np.sum(mask_fup) - np.sum(mask_base)
        _vol_fraction = np.sum(mask_fup) / np.sum(mask_base)
        _vol_diff_fraction = _vol_diff / np.sum(mask_base)

        tissue_metrics[tissue_id] = (_dice, _vol_diff, _vol_fraction, _vol_diff_fraction) + _jac_metrics + _div_metrics + _mag_metrics

    columns_names = ['dice', 'vol_diff', 'vol_fraction', 'vol_diff_fraction'] + get_colums_names(JACOBIAN) + get_colums_names(DIVERGENCE) + get_colums_names(MAGNITUDE) 
    tissue_metrics_df = pd.DataFrame.from_dict(tissue_metrics, orient='index', columns=columns_names)
    tissue_metrics_df.index.name = 'tissue_id'

    # # save results
    path_name = f'{current_path}/features/tissues_{nb_tissues}/{id}.csv'
    tissue_metrics_df.to_csv(path_name)

def compute_patient_metrics_for_all():
    matching_ids = pil.get_matching_ids()
    for nb_tissues in [3, 36, 96]:
        print(f'Computing metrics for {nb_tissues} tissues')
        for matching_id in tqdm(matching_ids):
            compute_patient_metrics(matching_id, nb_tissues=nb_tissues)

compute_patient_metrics_for_all()