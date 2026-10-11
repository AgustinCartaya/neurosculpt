import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../../'))
dfield_approach_elastix_path = os.path.abspath(os.path.join(current_path, '../'))

sys.path.append(base_path)
sys.path.append(dfield_approach_elastix_path)

import numpy as np
import pandas as pd
import project_images_loader as pil
import utils.util as util
import utils.nifti_functions as nfc
from tqdm import tqdm

import constants as cts
from similarity_tables_reader import SimilarityTablesReader




def compute_similar_images_mean_dfield(sim_metric, nb_images, df_metrics, pfile_number=cts.PFILE_0038):
    df_metrics = df_metrics.sort_values(sim_metric, ascending=False)
    df_best_metrics = df_metrics.head(nb_images)

    df_best_metrics['weight'] = df_best_metrics[sim_metric] / df_best_metrics[sim_metric].sum()
    dfield_sum = None

    progress_bar = tqdm(total=nb_images)
    for index, row in df_best_metrics.iterrows():
        id_value, weight = row['id'], row['weight']
        _dfield, affine = nfc.load_nifti(cts.get_image_path_name(id_value, pfile_number, iname=cts.INAME_DFIELD, subfolder=cts.FOLDER_REGISTRATIONS))
        if dfield_sum is None:
            dfield_sum = _dfield * weight
        else:
            dfield_sum += _dfield * weight
        progress_bar.update(1)
    progress_bar.close()

    return dfield_sum.astype(np.float32), affine


def create_similarity_dfields(id_indexs):
    matchin_ids = pil.get_matching_ids()
    sim_table_reder = SimilarityTablesReader(matchin_ids)

    # # compute mean dfield
    for id_index in id_indexs:
        id = matchin_ids[id_index] 
        for nb_images in [10, 100]:
            for sim_metric in ['ssim', 'dice_mean']:
                patient_sim_table = sim_table_reder.get_patient_similarity_table(id)
                dfield_mean, affine = compute_similar_images_mean_dfield(sim_metric, nb_images, patient_sim_table, pfile_number=cts.PFILE_0038)
                nfc.save_nifti(dfield_mean, affine, cts.get_similar_mean_dfield_path_name(id, sim_metric.upper(), cts.PFILE_0038, nb_images, subfolder=cts.FOLDER_SIMILAR_MEAN_DFIELDS))

# create_similarity_dfields([301, 302, 303, 304, 305, 306, 307])
create_similarity_dfields([300])