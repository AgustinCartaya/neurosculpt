"""Cross-sectional tissue features: volume (voxel count) of every tissue at HUNT3 and HUNT4."""
import numpy as np
import pandas as pd
from tqdm import tqdm

from brainpred import splits
from brainpred.features.common import TissueTableLoader, get_features_folder, load_segmentation, save_table


def get_features_path(nb_tissues):
    return get_features_folder("cross_sectional", "tissue_based", f"tissues_{nb_tissues}")


def compute_patient_metrics(id, hunt, nb_tissues):
    seg = load_segmentation(id, hunt, nb_tissues=nb_tissues, remove_cerebellum=(nb_tissues == 3))

    seg_indexs = np.sort(np.unique(seg.astype(np.uint)))
    tissue_metrics = {}
    for tissue_id in seg_indexs:
        tissue_metrics[tissue_id] = np.sum(seg == tissue_id)
    return tissue_metrics


def concat_dicts(dict_hunt3, dict_hunt4):
    return {key: (val_hunt3, dict_hunt4[key]) for key, val_hunt3 in dict_hunt3.items()}


def compute_patient_metrics_for_all(nb_tissues):
    matching_ids = splits.get_matching_ids()

    metrics = ['vol']
    columns_names = [f'{metric}_hunt3' for metric in metrics] + [f'{metric}_hunt4' for metric in metrics]

    print(f'Computing metrics for {nb_tissues} tissues')
    for id in tqdm(matching_ids):
        tissue_metrics_list = [compute_patient_metrics(id, hunt, nb_tissues=nb_tissues) for hunt in splits.HUNTS]
        patient_metrics = concat_dicts(tissue_metrics_list[0], tissue_metrics_list[1])

        tissue_metrics_df = pd.DataFrame.from_dict(patient_metrics, orient='index', columns=columns_names)
        tissue_metrics_df.index.name = 'tissue_id'
        save_table(tissue_metrics_df, f'{get_features_path(nb_tissues)}/{id}.csv', index=True)


class FeaturesLoader(TissueTableLoader):
    def __init__(self, nb_tissues, ids):
        super().__init__(get_features_path(nb_tissues), ids)
