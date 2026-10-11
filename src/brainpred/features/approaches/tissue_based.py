"""Tissue features of a method's postprocessed predictions: tissue volume and Dice against the real HUNT3 and HUNT4."""
import numpy as np
import pandas as pd
from tqdm import tqdm

from brainpred.features.approaches.common import get_approach_ids, load_result_image
from brainpred.features.common import TissueTableLoader, get_features_folder, load_segmentation_from_96, merge_segmentation_96, save_table
from brainpred.utils import metrics


def get_features_path(approach, nb_tissues):
    return get_features_folder("approaches", "tissue_based", approach, f"tissues_{nb_tissues}")


def compute_patient_metrics(id, nb_tissues, approach, extra_info):
    seg_hunt3 = load_segmentation_from_96(id, 3, nb_tissues=nb_tissues)
    seg_hunt4 = load_segmentation_from_96(id, 4, nb_tissues=nb_tissues)
    seg_result = merge_segmentation_96(load_result_image(id, approach, 'SEG_96', extra_info), nb_tissues)

    seg_indexs = np.sort(np.unique(seg_hunt4.astype(np.uint)))
    tissue_metrics = {}
    for tissue_id in seg_indexs:
        mask_hunt3 = seg_hunt3 == tissue_id
        mask_hunt4 = seg_hunt4 == tissue_id
        mask_result = seg_result == tissue_id

        _dice_hunt3 = metrics.compute_dice_coefficient(mask_result, mask_hunt3)
        _dice_hunt4 = metrics.compute_dice_coefficient(mask_result, mask_hunt4)
        _vol = np.sum(mask_result)

        tissue_metrics[tissue_id] = (_vol, _dice_hunt3, _dice_hunt4)

    columns_names = ['vol', 'dice_hunt3', 'dice_hunt4']
    tissue_metrics_df = pd.DataFrame.from_dict(tissue_metrics, orient='index', columns=columns_names)
    tissue_metrics_df.index.name = 'tissue_id'
    save_table(tissue_metrics_df, f'{get_features_path(approach, nb_tissues)}/{id}.csv', index=True)


def compute_patient_metrics_for_all(approach, extra_info, nb_tissues_list=(3, 36, 96)):
    print(f"Computing metrics for approach {approach} and extra_info {extra_info}")
    val_images = get_approach_ids(approach)
    for nb_tissues in nb_tissues_list:
        print(f'Computing metrics for {nb_tissues} tissues')
        for matching_id in tqdm(val_images):
            compute_patient_metrics(matching_id, nb_tissues, approach, extra_info)


class FeaturesLoader(TissueTableLoader):
    def __init__(self, approach, nb_tissues, ids):
        super().__init__(get_features_path(approach, nb_tissues), ids)
