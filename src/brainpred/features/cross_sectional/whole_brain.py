"""Cross-sectional whole-brain features: brain parenchymal fraction (BPF) at HUNT3 and HUNT4."""
import numpy as np
import pandas as pd
from tqdm import tqdm

from brainpred import paths, splits
from brainpred.features.common import WHOLE_BRAIN_TABLE_NAME, WholeBrainTableLoader, get_features_folder, save_table
from brainpred.utils import brain_metrics, io


def get_table_path_name():
    return f'{get_features_folder("cross_sectional", "whole_brain")}/{WHOLE_BRAIN_TABLE_NAME}.csv'


def load_seg_and_mask(id, hunt):
    seg, _ = io.load_nifti(paths.get_segmentation_path_name(hunt, id, 3))
    bmask_ncereb, _ = io.load_nifti(paths.get_brain_mask_path_name(hunt, id, paths.REGISTRATION_TO_MNI, no_cerebellum=True))
    return seg, bmask_ncereb


def compute_whole_brain_patient_statistics(id):
    seg_hunt3, bmask_ncereb_hunt3 = load_seg_and_mask(id, 3)
    seg_hunt4, _ = load_seg_and_mask(id, 4)
    # both BPFs are computed with the HUNT3 brain mask (without cerebellum)
    bpf_hunt3 = brain_metrics.compute_bpf(seg_hunt3, bmask_ncereb_hunt3)
    bpf_hunt4 = brain_metrics.compute_bpf(seg_hunt4, bmask_ncereb_hunt3)
    return (bpf_hunt3, bpf_hunt4)


def compute_patient_metrics_for_all():
    matching_ids = splits.get_matching_ids()

    metrics = ['bpf']
    columns_names = ['patient_id'] + [f'{metric}_hunt3' for metric in metrics] + [f'{metric}_hunt4' for metric in metrics]

    bpf_np = np.zeros((len(matching_ids), len(columns_names)))
    for i, id in enumerate(tqdm(matching_ids)):
        bpf_np[i, 0] = int(id)
        bpf_np[i, 1:] = list(compute_whole_brain_patient_statistics(id))

    save_table(pd.DataFrame(bpf_np, columns=columns_names), get_table_path_name())


class FeaturesLoader(WholeBrainTableLoader):
    def __init__(self):
        super().__init__(get_table_path_name())
