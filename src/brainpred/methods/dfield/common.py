"""Shared helpers of the deformation-field methods: reference selection and weighted dfield averaging."""
import numpy as np
from tqdm import tqdm

from brainpred import paths, splits
from brainpred.utils import io


def get_excluded_reference_ids(target_ids):
    """Patients never used as reference when predicting ``target_ids``: the targets, the validation set and the excluded patients."""
    return list(dict.fromkeys(list(target_ids) + splits.get_validation_ids() + splits.get_excluded_patients()))


def load_longitudinal_dfield(id):
    """HUNT3 -> HUNT4 elastix deformation field of patient ``id``, shape (x, y, z, 1, 3)."""
    return io.load_nifti(paths.get_longitudinal_registration_path_name(id, type_img="DFIELD"))


def compute_weighted_mean_dfield(df_best_metrics, load_dfield_fn):
    """Sum of ``weight * dfield`` over the rows (``id``, ``weight``) of ``df_best_metrics``."""
    dfield_sum = None
    affine = None
    progress_bar = tqdm(total=len(df_best_metrics))
    for _, row in df_best_metrics.iterrows():
        id, weight = row['id'], row['weight']
        _dfield, affine = load_dfield_fn(id)
        if dfield_sum is None:
            dfield_sum = _dfield * weight
        else:
            dfield_sum += _dfield * weight
        progress_bar.update(1)
    progress_bar.close()
    return dfield_sum.astype(np.float32), affine


# ---- region-wise (tissue / patch) selection
def convert_patient_similarity_table(inter_patient_table, column_names):
    """(n_refs, n_regions, 2) array of (reference index, similarity) and the reference ids as 5-digit strings."""
    dice_table = inter_patient_table[column_names].values

    parallel_ids = inter_patient_table["id"].values.astype(int)
    parallel_ids = ["{0:05}".format(parallel_id) for parallel_id in parallel_ids]

    similarity_table = np.zeros((len(parallel_ids), len(column_names), 2))
    for i in range(len(parallel_ids)):
        similarity_table[i, :, 0] = i
        similarity_table[i, :, 1] = dice_table[i]
    return similarity_table, parallel_ids


def sort_array_by_columns(array):
    """Sort each column (region) individually by decreasing similarity (``array[:, :, 1]``)."""
    sorted_indices = np.argsort(array[:, :, 1], axis=0)[::-1]
    return array[sorted_indices, np.arange(array.shape[1]), :]


def select_most_similar(similarity_table, nb_best_images):
    """The ``nb_best_images`` best references of every region, with their L1-normalised similarity
    appended as weight: (nb_best_images, n_regions, 3) array of (reference index, similarity, weight)."""
    similarity_table = sort_array_by_columns(similarity_table)
    most_similar = similarity_table[:nb_best_images]

    l1_norm = np.linalg.norm(most_similar[:, :, 1], ord=1, axis=0)
    l1_norm[l1_norm == 0] = 1
    weights_columns = most_similar[:, :, 1] / l1_norm
    return np.concatenate((most_similar, weights_columns[:, :, np.newaxis]), axis=2)
