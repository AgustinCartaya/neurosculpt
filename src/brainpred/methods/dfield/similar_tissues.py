"""SIM_TIS method: the target dfield is built tissue by tissue from the references with the highest Dice for that tissue.

``similar_tissues`` ignores the background (tissue 0); ``similar_tissues_v2`` keeps it.
"""
import os

import numpy as np
from tqdm import tqdm

from brainpred import paths, splits
from brainpred.features.inter_patient import tissue_based as inter_patient_tissue_based
from brainpred.methods.dfield import apply
from brainpred.methods.dfield.common import (convert_patient_similarity_table, get_excluded_reference_ids,
                                             load_longitudinal_dfield, select_most_similar)
from brainpred.results import ResultsLayout
from brainpred.utils import io

DFIELD_NAME_PATTERN = "{0}_TISSUESIM_DFIELD_{1}_{2}.nii.gz"
LAYOUTS = {
    "similar_tissues": ResultsLayout("dfield_similar_tissues", DFIELD_NAME_PATTERN),
    "similar_tissues_v2": ResultsLayout("dfield_similar_tissues_v2", DFIELD_NAME_PATTERN),
}
# tissues removed from the construction (v1: background; v2: none)
REMOVE_TISSUES = {"similar_tissues": [0], "similar_tissues_v2": None}


def compute_tissue_based_dfield(similarity_tissue_table, parallel_ids, seg_target, tissue_ids, nb_best_images=10):
    # 1-2. best references of each tissue and their weights (L1-normalised Dice)
    most_similar_tissues = select_most_similar(similarity_tissue_table, nb_best_images)

    # 3. dfields to load
    dfields_data_to_load = [int(i) for i in np.unique(most_similar_tissues[:, :, 0])]

    # 4. weighted mean of the reference dfields inside each target tissue
    img_size = seg_target.shape
    dfield_constructed = np.zeros((*img_size, 3))
    affine = None
    print("constructing target dfield from the best dfields")
    dfield_data_masked = np.zeros((*img_size, 3))
    # loop over the reference dfields (each loaded once)
    for img_data_index in tqdm(dfields_data_to_load):
        dfield_data, affine = load_longitudinal_dfield(parallel_ids[img_data_index])
        dfield_data = dfield_data[:, :, :, 0]
        for tissue_index in range(most_similar_tissues.shape[1]):
            current_tissue_table = most_similar_tissues[:, tissue_index, :]
            img_indices = current_tissue_table[:, 0]
            img_weights = current_tissue_table[:, 2]

            for index in np.where(img_indices == img_data_index)[0]:
                tissue_mask = seg_target == tissue_ids[tissue_index]
                dfield_data_masked *= 0
                dfield_data_masked[tissue_mask] = dfield_data[tissue_mask]
                dfield_constructed += img_weights[index] * dfield_data_masked

    dfield_constructed = np.expand_dims(dfield_constructed, axis=-2)
    return dfield_constructed, affine


def create_dfields(ids, nb_tissues=96, nb_images_list=(10,), variant="similar_tissues"):
    layout = LAYOUTS[variant]
    remove_ids = REMOVE_TISSUES[variant]
    dice_table_reader = inter_patient_tissue_based.FeaturesLoader(nb_tissues, splits.get_matching_ids())
    exclude_ids = get_excluded_reference_ids(ids)

    for id in ids:
        print(f"Processing {id}, nb_tissues: {nb_tissues}")
        seg_target = io.load_nifti(paths.get_segmentation_path_name(3, id, nb_tissues=nb_tissues))[0].astype(np.uint32)
        tissue_ids = np.unique(seg_target)
        if remove_ids is not None:
            tissue_ids = np.setdiff1d(tissue_ids, np.array(remove_ids))
        for nb_images in nb_images_list:
            dfield_path_name = layout.get_dfield_path_name(id, nb_tissues, nb_images)
            if not io.verify_file(dfield_path_name):
                if io.verify_folder(os.path.dirname(dfield_path_name), create=True):
                    df_patient_sim_table = dice_table_reader.get_patient_similarity_table(id, exclude_ids=exclude_ids)
                    similarity_tissue_table, parallel_ids = convert_patient_similarity_table(df_patient_sim_table, [f"dice_{i}" for i in tissue_ids])
                    dfield_mean, affine = compute_tissue_based_dfield(similarity_tissue_table, parallel_ids, seg_target, tissue_ids, nb_images)
                    io.save_nifti(dfield_mean, affine, dfield_path_name)


def apply_dfields(ids, nb_tissues=96, nb_images_list=(10,), variant="similar_tissues", img_types=tuple(apply.IMAGE_TYPES), skip_existing=False):
    layout = LAYOUTS[variant]
    for nb_images in nb_images_list:
        print(f"-------- Registering images using DFIELD {nb_tissues} with {nb_images} images")
        extra_info = (nb_tissues, nb_images)
        apply.apply_all(layout, ids, lambda id: layout.get_dfield_path_name(id, *extra_info), extra_info, img_types, skip_existing)
