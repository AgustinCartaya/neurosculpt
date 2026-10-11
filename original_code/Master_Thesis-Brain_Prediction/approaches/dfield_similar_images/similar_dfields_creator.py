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




def compute_similar_images_mean_dfield(sim_metric, nb_images, df_metrics):
    df_metrics = df_metrics.sort_values(sim_metric, ascending=False)
    df_best_metrics = df_metrics.head(nb_images).copy()

    df_best_metrics['weight'] = df_best_metrics[sim_metric] / df_best_metrics[sim_metric].sum()
    
    dfield_sum = None

    # print(df_best_metrics["id"])
    progress_bar = tqdm(total=nb_images)
    for index, row in df_best_metrics.iterrows():
        id, weight = row['id'], row['weight']
        _dfield, affine = nfc.load_nifti(pil.get_elastix_registered_image_path_name(id, type_img="DFIELD", absolute=True))
        if dfield_sum is None:
            dfield_sum = _dfield * weight
        else:
            dfield_sum += _dfield * weight
        progress_bar.update(1)
    progress_bar.close()

    return dfield_sum.astype(np.float32), affine


def create_similarity_dfields(ids):
    matching_ids = pil.get_matching_ids()

    ssim_table_reder = features_loader_whole_brain_based.FeaturesLoader(matching_ids)
    dice_mean_table_reder = features_loader_tissue_based.FeaturesLoader(3, matching_ids)

    # # compute mean dfield
    for id in ids:
        print("id", id)
        for nb_images in [5, 10, 100]:
            for sim_metric in ['ssim', 'dice_mean']:
                dfield_path_name = cts.get_dfield_path_name(id, sim_metric.upper(), nb_images)
                if not fc.verify_file(dfield_path_name):
                    if fc.verify_folder(os.path.dirname(dfield_path_name), create=True):
                        if sim_metric == 'ssim':
                            df_patient_sim_table = ssim_table_reder.get_patient_similarity_table(id, exclude_ids=ids + pil.EXCLUDED_PATIENTS)
                        elif sim_metric == 'dice_mean':
                            df_patient_sim_table = dice_mean_table_reder.get_patient_similarity_table(id, exclude_ids=ids + pil.EXCLUDED_PATIENTS)
                        print("df_patient_sim_table", df_patient_sim_table.shape)
                        dfield_mean, affine = compute_similar_images_mean_dfield(sim_metric, nb_images, df_patient_sim_table)

                        nfc.save_nifti(dfield_mean, affine, dfield_path_name)

# create_similarity_dfields([301, 302, 303, 304, 305, 306, 307])
val_images = ["08202"] + pil.get_matching_ids()[621:]
create_similarity_dfields(val_images)


