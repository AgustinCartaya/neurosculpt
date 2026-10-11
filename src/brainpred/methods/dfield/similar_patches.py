"""SIM_PATCH method: the target dfield is built patch by patch from the references with the most similar segmentation patch."""
import os

import numpy as np
from tqdm import tqdm

from brainpred.features.inter_patient import patch_based as inter_patient_patch_based
from brainpred.methods.dfield import apply
from brainpred.methods.dfield.common import (convert_patient_similarity_table, get_excluded_reference_ids,
                                             load_longitudinal_dfield, select_most_similar)
from brainpred.results import ResultsLayout
from brainpred.utils import io, patches

LAYOUT = ResultsLayout("dfield_similar_patches", "{0}_PATCHSIM_DFIELD_{1}_{2}.nii.gz")
NB_PATCHES = 1573  # patches of 32^3 with step 16 in the MNI volume


def compute_patch_based_dfield(similarity_patches_table, parallel_ids, nb_best_images=10,
                               patch_shape=inter_patient_patch_based.PATCH_SHAPE, extraction_step=inter_patient_patch_based.EXTRACTION_STEP):
    # 1-2. best references of each patch and their weights (L1-normalised similarity)
    most_similar_patches = select_most_similar(similarity_patches_table, nb_best_images)

    # 3. dfields to load
    dfields_data_to_load = [int(i) for i in np.unique(most_similar_patches[:, :, 0]) if i >= 0]

    # 4. weighted mean of the reference dfield patches
    patches_shape = (similarity_patches_table.shape[1], *patch_shape)
    patches_dfield_target_x = np.zeros(patches_shape)
    patches_dfield_target_y = np.zeros(patches_shape)
    patches_dfield_target_z = np.zeros(patches_shape)

    affine = None
    print("constructing target dfield from the best dfields")
    # loop over the reference dfields (each loaded once)
    for img_data_index in tqdm(dfields_data_to_load):
        dfield_data, affine = load_longitudinal_dfield(parallel_ids[img_data_index])
        dfield_data = dfield_data[:, :, :, 0]

        patches_dfield_data_x = patches.extract_patches(dfield_data[..., 0], patch_shape, extraction_step)
        patches_dfield_data_y = patches.extract_patches(dfield_data[..., 1], patch_shape, extraction_step)
        patches_dfield_data_z = patches.extract_patches(dfield_data[..., 2], patch_shape, extraction_step)

        for patch_index in range(most_similar_patches.shape[1]):
            # references contributing to the weighted mean of the current patch
            for img_index, img_dice, img_weight in most_similar_patches[:, patch_index, :]:
                if img_index == img_data_index:
                    patches_dfield_target_x[patch_index] += img_weight * patches_dfield_data_x[patch_index]
                    patches_dfield_target_y[patch_index] += img_weight * patches_dfield_data_y[patch_index]
                    patches_dfield_target_z[patch_index] += img_weight * patches_dfield_data_z[patch_index]

    # 5. reconstruct the dfield from the patches
    dfield_shape = dfield_data.shape[:3]
    dfield_target_x = patches.perform_voting(patches_dfield_target_x, patch_shape, dfield_shape, extraction_step, extraction_step)
    dfield_target_y = patches.perform_voting(patches_dfield_target_y, patch_shape, dfield_shape, extraction_step, extraction_step)
    dfield_target_z = patches.perform_voting(patches_dfield_target_z, patch_shape, dfield_shape, extraction_step, extraction_step)

    dfield_target = np.stack((dfield_target_x, dfield_target_y, dfield_target_z), axis=-1)
    dfield_target = np.expand_dims(dfield_target, axis=-2)
    return dfield_target, affine


def create_dfields(ids, nb_tissues=96, nb_images_list=(10,)):
    exclude_ids = get_excluded_reference_ids(ids)
    patches_table_reader = inter_patient_patch_based.FeaturesLoader(nb_tissues)
    for id in tqdm(ids):
        print(f"id {id} tissue {nb_tissues}")
        for nb_images in nb_images_list:
            dfield_path_name = LAYOUT.get_dfield_path_name(id, nb_tissues, nb_images)
            if not io.verify_file(dfield_path_name):
                if io.verify_folder(os.path.dirname(dfield_path_name), create=True):
                    df_patient_sim_table = patches_table_reader.get_patient_table(id, exclude_ids=exclude_ids)
                    similarity_patches_table, parallel_ids = convert_patient_similarity_table(df_patient_sim_table, [f"patch_{i}" for i in range(NB_PATCHES)])
                    dfield_mean, affine = compute_patch_based_dfield(similarity_patches_table, parallel_ids, nb_images)
                    io.save_nifti(dfield_mean, affine, dfield_path_name)


def apply_dfields(ids, nb_tissues=96, nb_images_list=(10,), img_types=tuple(apply.IMAGE_TYPES), skip_existing=False):
    for nb_images in nb_images_list:
        print(f"-------- Registering images using DFIELD {nb_tissues} with {nb_images} images")
        extra_info = (nb_tissues, nb_images)
        apply.apply_all(LAYOUT, ids, lambda id: LAYOUT.get_dfield_path_name(id, *extra_info), extra_info, img_types, skip_existing)
