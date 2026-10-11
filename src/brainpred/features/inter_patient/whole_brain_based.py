"""Inter-patient whole-brain image similarity (NCC, SSIM between skull-stripped MNI images)."""
import numpy as np
import pandas as pd
from tqdm import tqdm

from brainpred import paths, splits
from brainpred.features.common import PairwiseSimilarityLoader, get_features_folder, save_table
from brainpred.utils import intensity, io, metrics


def get_features_path(hunt=3):
    return get_features_folder("inter_patient", "whole_brain_based", f"features_hunt{hunt}")


def load_img_mask_and_skull_stripping(id, hunt, nb_tissues=3):
    img_target, _ = io.load_nifti(paths.get_registered_to_template_image_path_name(hunt, id, paths.REGISTRATION_TO_MNI))
    mask_target, _ = io.load_nifti(paths.get_segmentation_path_name(hunt, id, nb_tissues))
    return intensity.skull_stripping_and_normalize(img_target, mask_target)


def get_img_sim_metrics(img_f, img_m):
    ncc = metrics.compute_normalized_cross_correlation(img_f, img_m)
    ssim = metrics.compute_ssim(img_f, img_m)
    return [ncc, ssim]


def create_similarity_matrix(hunt=3):
    """Write ``<id>.csv`` with the similarity of ``id`` against every later patient (upper triangle)."""
    matchin_ids = splits.get_matching_ids()
    nb_ids = len(matchin_ids)

    columns = ['id', 'ncc', 'ssim']
    nb_columns = len(columns)
    for index_target, id_target in enumerate(matchin_ids[:-1]):
        print(f"Processing {id_target} ({index_target+1}/{nb_ids-1})")

        df_path_name = f'{get_features_path(hunt)}/{id_target}.csv'
        if io.verify_file(df_path_name):
            continue

        img_target = load_img_mask_and_skull_stripping(id_target, hunt)
        similarity_matrix = np.zeros((nb_ids-index_target-1, nb_columns))
        for i, id_candidate in enumerate(tqdm(matchin_ids[index_target+1:])):
            img_candidate = load_img_mask_and_skull_stripping(id_candidate, hunt)
            metrics_image = np.array(get_img_sim_metrics(img_target, img_candidate))
            similarity_matrix[i] = np.concatenate((np.array([id_candidate]), metrics_image))

        save_table(pd.DataFrame(similarity_matrix, columns=columns), df_path_name)


class FeaturesLoader(PairwiseSimilarityLoader):
    def __init__(self, ids, hunt=3):
        super().__init__(get_features_path(hunt), ids, allow_missing=True)
