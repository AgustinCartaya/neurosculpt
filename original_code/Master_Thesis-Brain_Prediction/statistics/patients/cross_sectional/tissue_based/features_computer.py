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
# import metrics_computer as metrics_computer
import utils.util_freesurfer_segmentation as ufs


# np.set_printoptions(precision=4)

# JACOBIAN = 'jac'
# DIVERGENCE = 'div'
# MAGNITUDE = 'mag'

# def get_colums_names(att):
#     return [f'{att}_min', f'{att}_max', f'{att}_mean', f'{att}_std']


def load_segmentation(id, hunt, nb_tissues, remove_cerebellum=False):
    seg, _ = nfc.load_nifti(pil.get_segmentation_path_name(hunt, id, nb_tissues, absolute=True))
    if remove_cerebellum:
        seg_36, _ = nfc.load_nifti(pil.get_segmentation_path_name(hunt, id, 36, absolute=True))
        return ufs.remove_cerebellum(seg, seg_36, deep_copy=False)
    return seg

def compute_patient_metrics(id, hunt, nb_tissues):
    seg = load_segmentation(id, hunt, nb_tissues=nb_tissues, remove_cerebellum=(nb_tissues==3))
    
    seg_indexs = np.unique(seg.astype(np.uint))
    seg_indexs = np.sort(seg_indexs)

    tissue_metrics = {}
    for tissue_id in np.unique(seg_indexs):
        mask = seg == tissue_id
        _vol = np.sum(mask)

        tissue_metrics[tissue_id] = _vol

    return tissue_metrics


def concat_dicts(dict_hunt3, dict_hunt4):
    new_dict = {}
    for key, val_hunt3 in dict_hunt3.items():
        new_dict[key] = (val_hunt3, dict_hunt4[key])
    return new_dict

def compute_patient_metrics_for_all(nb_tissues):
    matching_ids = pil.get_matching_ids()

    metrics = ['vol']
    metrics_hunt3 = [ f'{metric}_hunt3' for metric in metrics]
    metrics_hunt4 = [ f'{metric}_hunt4' for metric in metrics]
    columns_names = metrics_hunt3 + metrics_hunt4


    print(f'Computing metrics for {nb_tissues} tissues')
    for id in tqdm(matching_ids):
        tissue_metrics_list = []
        for hunt in [3, 4]:
            tissue_metrics_list.append(compute_patient_metrics(id, hunt, nb_tissues=nb_tissues))
        patient_metrics = concat_dicts(tissue_metrics_list[0], tissue_metrics_list[1])


        # save results
        tissue_metrics_df = pd.DataFrame.from_dict(patient_metrics, orient='index', columns=columns_names)
        tissue_metrics_df.index.name = 'tissue_id'

        path_name = f'{current_path}/features/tissues_{nb_tissues}/{id}.csv'
        tissue_metrics_df.to_csv(path_name)


compute_patient_metrics_for_all(3)
# compute_patient_metrics_for_all(36)
# compute_patient_metrics_for_all(96)