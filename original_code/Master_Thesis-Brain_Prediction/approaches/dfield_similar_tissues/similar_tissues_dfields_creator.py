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
from tqdm import tqdm

import constants as cts
# from similarity_tables_reader import SimilarityTablesReader
features_loader_tissue_based = fc.load_module_from_path('features_loader_tissue_based', os.path.join(base_path, 'statistics/patients/inter_patient/tissue_based/features_loader.py'))




def sort_array_by_columns(array):
    """
    Sorts each column (3rd dimension) individually of a 3-dimensional array.
    """
    ## Fats way to sort the array by columns
    sorted_indices = np.argsort(array[:, :, 1], axis=0)[::-1]
    array_sorted = array[sorted_indices, np.arange(array.shape[1]), :]
    return array_sorted


def compute_tissue_based_dfield(similarity_tissue_table, parallel_ids, seg_target, tissue_ids, nb_best_images=10):
   
    # 1. sort patches_dice by columns individually to obtain the best image for each patch
    similarity_tissue_table = sort_array_by_columns(similarity_tissue_table)
    most_similar_tissues = similarity_tissue_table[:nb_best_images]

    # 2. compute best patches weights (l1 normalized dice scores)
    l1_norm = np.linalg.norm(most_similar_tissues[:, :, 1], ord=1, axis=0)
    l1_norm[l1_norm == 0] = 1
    weights_columns = most_similar_tissues[:,:,1] / l1_norm
    most_similar_tissues = np.concatenate((most_similar_tissues, weights_columns[:, :, np.newaxis]), axis=2)

    # 3. ontain dfields index to load
    dfields_data_to_load = np.unique(most_similar_tissues[:, :, 0])
    dfields_data_to_load = [int(i) for i in dfields_data_to_load]

    # 4. compute dfield patch weighted mean
    img_size = seg_target.shape
    dfield_constructed = np.zeros((*img_size, 3))

    affine = None
    # 5. recontruct the dfield from the patches
    # looping over the best dfields in order to be faster
    print("constructing target dfield from the best dfields")
    dfield_data_masked = np.zeros((*img_size, 3))
    for img_data_index in tqdm(dfields_data_to_load):
        id_data = parallel_ids[img_data_index]
        dfield_data, affine = nfc.load_nifti(pil.get_elastix_registered_image_path_name(id_data, type_img="DFIELD", absolute=True))
        dfield_data = dfield_data[:,:,:,0]
        # loop over the tissues
        for tissue_index in range(most_similar_tissues.shape[1]):
            current_tissue_table = most_similar_tissues[:, tissue_index, :]
            img_indices = current_tissue_table[:, 0]
            img_weights = current_tissue_table[:, 2]

            matching_indices = np.where(img_indices == img_data_index)[0]
            # could be done with an if statement but it is faster this way
            for index in matching_indices:
                tissue_mask = seg_target == tissue_ids[tissue_index]
                dfield_data_masked *= 0
                dfield_data_masked[tissue_mask] = dfield_data[tissue_mask]

                dfield_constructed += img_weights[index] * dfield_data_masked
        
    dfield_constructed = np.expand_dims(dfield_constructed, axis=-2)
    return dfield_constructed, affine


def convert_patient_similarity_table_to_similarity_tissue_table(inter_patient_table, tissue_ids):
    column_names = [f"dice_{i}" for i in tissue_ids]
    dice_table = inter_patient_table[column_names].values

    parallel_ids = inter_patient_table["id"].values.astype(int)
    parallel_ids = ["{0:05}".format(parallel_id) for parallel_id in parallel_ids]

    similarity_tissue_table = np.zeros((len(parallel_ids), len(tissue_ids), 2))
    for i in range(len(parallel_ids)):
        similarity_tissue_table[i, :, 0] = i
        similarity_tissue_table[i, :, 1] = dice_table[i]
    return similarity_tissue_table, parallel_ids


def create_similarity_dfields(ids, nb_tissues, remove_ids=None):
    matching_ids = pil.get_matching_ids()
    dice_mean_table_reder = features_loader_tissue_based.FeaturesLoader(nb_tissues, matching_ids)

    # # compute mean dfield
    for id in ids:
        print(f"Processing {id}, nb_tissues: {nb_tissues}")
        seg_target = nfc.load_nifti(pil.get_segmentation_path_name(3, id, nb_tissues=nb_tissues, absolute=True))[0].astype(np.uint32)
        tissue_ids = np.unique(seg_target)
        if remove_ids is not None:
            tissue_ids = np.setdiff1d(tissue_ids, np.array(remove_ids))
        for nb_images in [10, 100]:
            dfield_path_name = cts.get_dfield_path_name(id, nb_tissues, nb_images)
            # if not fc.verify_file(dfield_path_name): # when finished uncomment this line
            if fc.verify_folder(os.path.dirname(dfield_path_name), create=True):
                df_patient_sim_table = dice_mean_table_reder.get_patient_similarity_table(id, exclude_ids=ids + pil.EXCLUDED_PATIENTS)
                similarity_tissue_table, parallel_ids = convert_patient_similarity_table_to_similarity_tissue_table(df_patient_sim_table, tissue_ids)
                dfield_mean, affine = compute_tissue_based_dfield(similarity_tissue_table, parallel_ids, seg_target, tissue_ids, nb_images)
                nfc.save_nifti(dfield_mean, affine, dfield_path_name)
        break

val_images = ["08202"] + pil.get_matching_ids()[621:]
# create_similarity_dfields(val_images, 3, remove_ids=[0])
# create_similarity_dfields(val_images, 96, remove_ids=[0])


