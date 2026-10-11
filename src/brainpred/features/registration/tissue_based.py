"""Registration tissue features: per-tissue Dice / volume change between the registered HUNT3 and the real HUNT4 segmentation."""
import numpy as np
import pandas as pd
from tqdm import tqdm

from brainpred import paths
from brainpred.features.common import TissueTableLoader, get_features_folder, save_table
from brainpred.utils import io, metrics
from brainpred.utils import segmentation as ufs


def get_features_path(nb_tissues):
    return get_features_folder("registration", "tissue_based", f"tissues_{nb_tissues}")


def load_segmentation(id, hunt, nb_tissues, remove_cerebellum=False):
    """HUNT3: 96-tissue segmentation registered to HUNT4; HUNT4: original 96-tissue segmentation."""
    if hunt == 3:
        seg, _ = io.load_nifti(paths.get_longitudinal_registration_path_name(id, "SEG_96"))
    else:
        seg, _ = io.load_nifti(paths.get_segmentation_path_name(hunt, id, nb_tissues=96))

    if remove_cerebellum:
        seg = ufs.remove_cerebellum(seg, seg, deep_copy=False)

    if nb_tissues == 3:
        seg = ufs.merge_seg96_to_seg3(seg)
    elif nb_tissues == 36:
        seg = ufs.merge_seg96_to_seg36(seg)
    return seg


def compute_patient_metrics(id, nb_tissues):
    seg_base = load_segmentation(id, 3, nb_tissues=nb_tissues, remove_cerebellum=(nb_tissues == 3))
    seg_fup = load_segmentation(id, 4, nb_tissues=nb_tissues, remove_cerebellum=(nb_tissues == 3))

    seg_indexs = np.sort(np.unique(np.concatenate([seg_base, seg_fup]).astype(np.uint)))
    tissue_metrics = {}
    for tissue_id in seg_indexs:
        mask_base = seg_base == tissue_id
        mask_fup = seg_fup == tissue_id

        _dice = metrics.compute_dice_coefficient(mask_base, mask_fup)
        _vol_diff = np.sum(mask_fup) - np.sum(mask_base)
        _vol_fraction = np.sum(mask_fup) / np.sum(mask_base)
        _vol_diff_fraction = _vol_diff / np.sum(mask_base)

        tissue_metrics[tissue_id] = (_dice, _vol_diff, _vol_fraction, _vol_diff_fraction)

    columns_names = ['dice', 'vol_diff', 'vol_fraction', 'vol_diff_fraction']
    tissue_metrics_df = pd.DataFrame.from_dict(tissue_metrics, orient='index', columns=columns_names)
    tissue_metrics_df.index.name = 'tissue_id'
    save_table(tissue_metrics_df, f'{get_features_path(nb_tissues)}/{id}.csv', index=True)


def compute_patient_metrics_for_all(ids, nb_tissues_list=(96,)):
    for nb_tissues in nb_tissues_list:
        print(f'Computing metrics for {nb_tissues} tissues')
        for matching_id in tqdm(ids):
            compute_patient_metrics(matching_id, nb_tissues=nb_tissues)


class FeaturesLoader(TissueTableLoader):
    def __init__(self, nb_tissues, ids):
        super().__init__(get_features_path(nb_tissues), ids)
