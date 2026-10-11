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
features_loader_whole_brain_based = fc.load_module_from_path('features_loader_whole_brain', os.path.join(base_path, 'statistics/patients/inter_patient/whole_brain_based/features_loader.py'))
features_loader_tissue_based = fc.load_module_from_path('features_loader_tissue_based', os.path.join(base_path, 'statistics/patients/inter_patient/tissue_based/features_loader.py'))




def compute_similar_images_mean_dfield(id_target, sim_metric, reg_type, nb_images, df_best_metrics):
    dfield_sum = None

    progress_bar = tqdm(total=nb_images)
    for index, row in df_best_metrics.iterrows():
        id, weight = row['id'], row['weight']
        _dfield, affine = nfc.load_nifti(cts.get_patient_registered_dfield_path_name(id_target, id, reg_type))
        if dfield_sum is None:
            dfield_sum = _dfield * weight
        else:
            dfield_sum += _dfield * weight
        progress_bar.update(1)
    progress_bar.close()

    return dfield_sum.astype(np.float32), affine


def create_similarity_dfields(ids):
    matching_ids = pil.get_matching_ids()

    dice_mean_table_reder = features_loader_tissue_based.FeaturesLoader(3, matching_ids)

    # # compute mean dfield
    for id in ids:
        print("id", id)
        for nb_images in [5]:
            # for reg_type in [cts.PARAMETERS_FILE_AFFINE, cts.PARAMETERS_FILE_BSPLINE_FAST]:
            for reg_type in [cts.PARAMETERS_FILE_BSPLINE_FAST]:
                dfield_path_name = cts.get_dfield_path_name(id, reg_type, nb_images)
                if not fc.verify_file(dfield_path_name):
                    if fc.verify_folder(os.path.dirname(dfield_path_name), create=True):
                        df_patient_sim_table = dice_mean_table_reder.get_sorted_similarity_table(id, "dice_mean", nb_images, exclude_ids= ids + pil.EXCLUDED_PATIENTS)
                        print("df_patient_sim_table", df_patient_sim_table.shape)
                        dfield_mean, affine = compute_similar_images_mean_dfield(id, "dice_mean", reg_type, nb_images, df_patient_sim_table)
                        nfc.save_nifti(dfield_mean, affine, dfield_path_name)


val_images = ["08202"] + pil.get_matching_ids()[621:]
create_similarity_dfields(val_images)


