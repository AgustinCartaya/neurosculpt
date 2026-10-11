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


def load_and_img(id, hunt):
    img, _ = nfc.load_nifti(pil.get_registered_to_template_image_path_name(hunt, id, registration_to=pil.REGISTRATION_TO_MNI, absolute=True))
    seg, _ = nfc.load_nifti(pil.get_segmentation_path_name(hunt, id, 3, absolute=True))
    bmask_ncereb, _ = nfc.load_nifti(pil.get_brain_mask_path_name(hunt, id, pil.REGISTRATION_TO_MNI, no_cerebellum=True, absolute=True))
    return img, seg, bmask_ncereb

def compute_whole_brain_patient_statistics(id):
    _, seg_hunt3, bmask_ncereb_hunt3 = load_and_img(id, 3)
    _, seg_hunt4, _ = load_and_img(id, 4)
    # compute bpf
    bpf_hunt3 = metrics_computer.compute_bpf(seg_hunt3, bmask_ncereb_hunt3)
    bpf_hunt4 = metrics_computer.compute_bpf(seg_hunt4, bmask_ncereb_hunt3)
    return (bpf_hunt3,bpf_hunt4)


def compute_patient_metrics_for_all():
    matching_ids = pil.get_matching_ids()

    metrics = ['bpf']
    metrics_hunt3 = [ f'{metric}_hunt3' for metric in metrics]
    metrics_hunt4 = [ f'{metric}_hunt4' for metric in metrics]
    columns_names = ['patient_id'] + metrics_hunt3 + metrics_hunt4

    bpf_np = np.zeros((len(matching_ids), len(columns_names)))

    for i, id in enumerate(tqdm(matching_ids)):
        patient_metrics = []
        patient_metrics.extend(list(compute_whole_brain_patient_statistics(id)))

        bpf_np[i, 0] = int(id)
        bpf_np[i, 1:] = patient_metrics
        
    df = pd.DataFrame(bpf_np, columns=columns_names)
    path_name = f'{current_path}/features/whole_brains_stats.csv'
    df.to_csv(path_name, index=False)

compute_patient_metrics_for_all()

