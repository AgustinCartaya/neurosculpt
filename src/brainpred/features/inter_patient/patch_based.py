"""Inter-patient patch similarity: tissue-weighted Dice of every 3D segmentation patch (validation vs reference patients)."""
import numpy as np
import pandas as pd
from tqdm import tqdm

from brainpred import paths
from brainpred.features.common import get_features_folder, save_table
from brainpred.utils import io, metrics, patches

PATCH_SHAPE = (32, 32, 32)
EXTRACTION_STEP = (16, 16, 16)


def get_features_path(nb_tissues):
    return get_features_folder("inter_patient", "patch_based", f"tissues_{nb_tissues}")


def compute_tissue_weigthed_dice(patch_target, patch_data):
    tissue_in_target, nb_tissue_voxels_in_target = np.unique(patch_target, return_counts=True)
    # NOTE: original behaviour kept: the "data" tissues are also counted on patch_target.
    tissue_in_data, nb_tissue_voxels_in_data = np.unique(patch_target, return_counts=True)

    tissue_volume_in_target = nb_tissue_voxels_in_target / np.prod(patch_target.shape)
    tissue_volume_in_data = nb_tissue_voxels_in_data / np.prod(patch_data.shape)

    weighted_sum = 0
    total_tissue_weight = 0
    for i, tissue_index in enumerate(tissue_in_target):
        if tissue_index in tissue_in_data:
            tissue_weight_target = tissue_volume_in_target[i]

            i_data = np.where(tissue_in_data == tissue_index)[0][0]
            tissue_weight_data = tissue_volume_in_data[i_data]

            mask1 = patch_target == tissue_index
            mask2 = patch_data == tissue_index

            tissue_dice = metrics.compute_dice_coefficient(mask1, mask2)
            tissue_weight = 0
            weighted_sum += (tissue_weight + tissue_weight_target + tissue_weight_data) * tissue_dice
            total_tissue_weight += tissue_weight

    dice = weighted_sum / (2+total_tissue_weight)
    return dice


def compute_patches_similarity_table(seg_target, ids_dataset, nb_tissues, patch_shape=PATCH_SHAPE, extraction_step=EXTRACTION_STEP):
    """Rows = reference patients (``id`` + one similarity per patch of ``seg_target``)."""
    patches_target = patches.extract_patches(seg_target, patch_shape, extraction_step)

    similarity_patches_table = np.zeros((len(ids_dataset), len(patches_target)+1))
    for img_index, id_data in enumerate(tqdm(ids_dataset)):
        mask_data, _ = io.load_nifti(paths.get_segmentation_path_name(3, id_data, nb_tissues=nb_tissues))
        patches_data = patches.extract_patches(mask_data, patch_shape, extraction_step)
        similarity_patches_table[img_index, 0] = int(id_data)
        for patch_index, patch_target in enumerate(patches_target):
            if np.any(patch_target > 0) and np.any(patches_data[patch_index] > 0):
                similarity_patches_table[img_index, patch_index+1] = compute_tissue_weigthed_dice(patch_target, patches_data[patch_index])
    return similarity_patches_table


def create_and_save_similarity_patches_tables(ids_val, ids_train, nb_tissues):
    """Write ``<id_val>.csv`` with the patch similarities of each validation patient against ``ids_train``."""
    for id_val in ids_val:
        patient_features_path = f'{get_features_path(nb_tissues)}/{id_val}.csv'
        if io.verify_file(patient_features_path):
            continue
        print(f"Processing {id_val} nb_tissues: {nb_tissues}")
        seg_target, _ = io.load_nifti(paths.get_segmentation_path_name(3, id_val, nb_tissues))
        similarity_patches_table = compute_patches_similarity_table(seg_target, ids_train, nb_tissues)

        columns = ["id"]+[f"patch_{i}" for i in range(similarity_patches_table.shape[1]-1)]
        save_table(pd.DataFrame(similarity_patches_table, columns=columns), patient_features_path)


class FeaturesLoader:
    def __init__(self, nb_tissues):
        self.path = get_features_path(nb_tissues)

    def get_patient_table(self, id, exclude_ids=None):
        df = self._load_csv(id)
        if exclude_ids is not None:
            exclude_ids = [int(id) for id in exclude_ids]
            df = df[~df['id'].isin(exclude_ids)]
        return df

    def _load_csv(self, id):
        return pd.read_csv(f'{self.path}/{id}.csv')
