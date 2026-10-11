import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))

sys.path.append(base_path)

import numpy as np
import pandas as pd
import project_images_loader as pil
import utils.util as util
import utils.functions as fc
import utils.nifti_functions as nfc
import utils.patches_controller as patches_controller

from tqdm import tqdm

import constants as cts

features_loader_patch_based = fc.load_module_from_path('features_loader_patch_based', os.path.join(base_path, 'statistics/patients/inter_patient/patch_based/features_loader.py'))

def sort_array_by_columns(array):
    """
    Sorts each column (3rd dimension) individually of a 3-dimensional array.
    """
    ## Fats way to sort the array by columns
    sorted_indices = np.argsort(array[:, :, 1], axis=0)[::-1]
    array_sorted = array[sorted_indices, np.arange(array.shape[1]), :]
    return array_sorted

def compute_pathc_based_dfield(similarity_patches_table, parallel_ids, nb_best_images=10, patch_shape=(32,32,32), extraction_step=(16,16,16)):
    # 1. sort patches_dice by columns individually to obtain the best image for each patch
    similarity_patches_table = sort_array_by_columns(similarity_patches_table)
    most_similar_patches = similarity_patches_table[:nb_best_images]

    # 2. compute best patches weights (l1 normalized dice scores)
    l1_norm = np.linalg.norm(most_similar_patches[:, :, 1], ord=1, axis=0)
    l1_norm[l1_norm == 0] = 1
    weights_columns = most_similar_patches[:,:,1] / l1_norm
    most_similar_patches = np.concatenate((most_similar_patches, weights_columns[:, :, np.newaxis]), axis=2)

    # 3. ontain dfields index to load
    dfields_data_to_load = np.unique(most_similar_patches[:, :, 0])
    dfields_data_to_load = [int(i) for i in dfields_data_to_load if i >= 0]

    # 4. compute dfield patch weighted mean
    pathces_shape = (similarity_patches_table.shape[1], *patch_shape)
    patches_dfield_target_x = np.zeros(pathces_shape)
    patches_dfield_target_y = np.zeros(pathces_shape)
    patches_dfield_target_z = np.zeros(pathces_shape)
    
    affine = None
    dfiled_shape = None
    # 5. recontruct the dfield from the patches
    # looping over the best dfields in order to be faster
    print("constructing target dfield from the best dfields")
    for img_data_index in tqdm(dfields_data_to_load):
        # dfield_data_path_name = cts.get_image_path_name(target_ids[img_data_index], cts.PFILE_0038, cts.INAME_DFIELD, subfolder=cts.FOLDER_REGISTRATIONS)
        # dfield_data, affine = nfc.load_nifti(dfield_data_path_name)
        id_data = parallel_ids[img_data_index]
        dfield_data, affine = nfc.load_nifti(pil.get_elastix_registered_image_path_name(id_data, type_img="DFIELD", absolute=True))
        dfield_data = dfield_data[:,:,:,0]
        
        dfield_data_x, dfield_data_y, dfield_data_z = dfield_data[..., 0], dfield_data[..., 1], dfield_data[..., 2]
        patches_dfield_data_x = patches_controller.extract_patches(dfield_data_x, patch_shape, extraction_step)
        patches_dfield_data_y = patches_controller.extract_patches(dfield_data_y, patch_shape, extraction_step)
        patches_dfield_data_z = patches_controller.extract_patches(dfield_data_z, patch_shape, extraction_step)

        # looping over the patches
        for patch_index in range(most_similar_patches.shape[1]):
            # looping over the dfields needed to compute the weighted mean for the current patch
            current_patch_table = most_similar_patches[:, patch_index, :]
            for img_index, img_dice, img_weight  in current_patch_table:
                # check the loaded best dfield is part of the weigthted mean of the current patch
                if img_index == img_data_index:
                    # add the dfield to the weighted mean of the current patch
                    patches_dfield_target_x[patch_index] += img_weight * patches_dfield_data_x[patch_index]
                    patches_dfield_target_y[patch_index] += img_weight * patches_dfield_data_y[patch_index]
                    patches_dfield_target_z[patch_index] += img_weight * patches_dfield_data_z[patch_index]

    # 7. Reoconstruct the dfield from the patches
    dfiled_shape = dfield_data.shape[:3]
                    
    dfield_target_x = patches_controller.perform_voting(patches_dfield_target_x, patch_shape, dfiled_shape, extraction_step, extraction_step)
    dfield_target_y = patches_controller.perform_voting(patches_dfield_target_y, patch_shape, dfiled_shape, extraction_step, extraction_step)
    dfield_target_z = patches_controller.perform_voting(patches_dfield_target_z, patch_shape, dfiled_shape, extraction_step, extraction_step)

    dfield_target = np.stack((dfield_target_x, dfield_target_y, dfield_target_z), axis=-1)
    dfield_target = np.expand_dims(dfield_target, axis=-2)
    return dfield_target, affine



def convert_patient_similarity_table_to_similarity_patches_table(inter_patient_table, nb_patches=1573):
    column_names = [f"patch_{i}" for i in range(nb_patches)]
    dice_table = inter_patient_table[column_names].values

    parallel_ids = inter_patient_table["id"].values.astype(int)
    parallel_ids = ["{0:05}".format(parallel_id) for parallel_id in parallel_ids]

    similarity_tissue_table = np.zeros((len(parallel_ids), len(column_names), 2))
    for i in range(len(parallel_ids)):
        similarity_tissue_table[i, :, 0] = i
        similarity_tissue_table[i, :, 1] = dice_table[i]
    return similarity_tissue_table, parallel_ids



def create_patch_based_similarity_dfields(ids, nb_tissues):
    
    # # compute mean dfield
    for id in tqdm(ids):
        print(f"id {id} tissue {nb_tissues}")
        # for nb_images in [10, 100]:
        for nb_images in [10]:
            dfield_path_name = cts.get_dfield_path_name(id, nb_tissues, nb_images)
            if not fc.verify_file(dfield_path_name):
                if fc.verify_folder(os.path.dirname(dfield_path_name), create=True):
                    patches_table_reder = features_loader_patch_based.FeaturesLoader(nb_tissues)
                    df_patient_sim_table = patches_table_reder.get_patient_table(id, exclude_ids=ids + pil.EXCLUDED_PATIENTS)
                    similarity_patches_table, parallel_ids = convert_patient_similarity_table_to_similarity_patches_table(df_patient_sim_table, nb_patches=1573)
                    dfield_mean, affine = compute_pathc_based_dfield(similarity_patches_table, parallel_ids, nb_images)

                    # nfc.save_nifti(dfield_mean, affine, dfield_path_name)

# create_similarity_dfields([301, 302, 303, 304, 305, 306, 307])
val_images = ["08202"] + pil.get_matching_ids()[621:]
create_patch_based_similarity_dfields(val_images, nb_tissues=3)
# create_patch_based_similarity_dfields(val_images, nb_tissues=96)


