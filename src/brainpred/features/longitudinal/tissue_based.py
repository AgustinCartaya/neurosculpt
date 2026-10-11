"""Longitudinal tissue features: per-tissue Dice / volume change and statistics of the HUNT3->HUNT4 deformation field."""
import numpy as np
import pandas as pd
from tqdm import tqdm

from brainpred import paths, splits
from brainpred.features.common import TissueTableLoader, get_features_folder, load_segmentation_from_96, save_table
from brainpred.utils import dfield, intensity, io, metrics

JACOBIAN = 'jac'
DIVERGENCE = 'div'
MAGNITUDE = 'mag'


def get_features_path(nb_tissues):
    return get_features_folder("longitudinal", "tissue_based", f"tissues_{nb_tissues}")


def get_colums_names(att):
    return [f'{att}_min', f'{att}_max', f'{att}_mean', f'{att}_std']


def compute_patient_metrics(id, nb_tissues):
    seg_base = load_segmentation_from_96(id, 3, nb_tissues=nb_tissues)
    seg_fup = load_segmentation_from_96(id, 4, nb_tissues=nb_tissues)

    dfield_np, dfield_sitk = io.load_image_sitk(paths.get_longitudinal_registration_path_name(id, "DFIELD"))

    tissue_jac = dfield.compute_dfield_jacobian(dfield_sitk)
    tissue_div = dfield.compute_dfield_divergence(dfield_np)
    tissue_mag = dfield.compute_dfield_magnitude(dfield_np)

    seg_indexs = np.sort(np.unique(seg_fup.astype(np.uint)))
    tissue_metrics = {}
    for tissue_id in seg_indexs:
        mask_base = seg_base == tissue_id
        mask_fup = seg_fup == tissue_id

        _jac_metrics = intensity.get_basic_metrics(tissue_jac[mask_base])
        _div_metrics = intensity.get_basic_metrics(tissue_div[mask_base])
        _mag_metrics = intensity.get_basic_metrics(tissue_mag[mask_base])

        _dice = metrics.compute_dice_coefficient(mask_base, mask_fup)
        _vol_diff = np.sum(mask_fup) - np.sum(mask_base)
        _vol_fraction = np.sum(mask_fup) / np.sum(mask_base)
        _vol_diff_fraction = _vol_diff / np.sum(mask_base)

        tissue_metrics[tissue_id] = (_dice, _vol_diff, _vol_fraction, _vol_diff_fraction) + _jac_metrics + _div_metrics + _mag_metrics

    columns_names = ['dice', 'vol_diff', 'vol_fraction', 'vol_diff_fraction'] + get_colums_names(JACOBIAN) + get_colums_names(DIVERGENCE) + get_colums_names(MAGNITUDE)
    tissue_metrics_df = pd.DataFrame.from_dict(tissue_metrics, orient='index', columns=columns_names)
    tissue_metrics_df.index.name = 'tissue_id'
    save_table(tissue_metrics_df, f'{get_features_path(nb_tissues)}/{id}.csv', index=True)


def compute_patient_metrics_for_all(nb_tissues_list=(3, 36, 96)):
    matching_ids = splits.get_matching_ids()
    for nb_tissues in nb_tissues_list:
        print(f'Computing metrics for {nb_tissues} tissues')
        for matching_id in tqdm(matching_ids):
            compute_patient_metrics(matching_id, nb_tissues=nb_tissues)


class FeaturesLoader(TissueTableLoader):
    def __init__(self, nb_tissues, ids):
        super().__init__(get_features_path(nb_tissues), ids)
