"""Inter-patient tissue similarity: per-tissue Dice and absolute volume difference between segmentations."""
import numpy as np
import pandas as pd
from tqdm import tqdm

from brainpred import splits
from brainpred.features.common import PairwiseSimilarityLoader, get_features_folder, load_segmentation, save_table
from brainpred.features.cross_sectional import tissue_based as cross_sectional_features
from brainpred.utils import io, metrics


def get_features_path(nb_tissues, hunt=3):
    return get_features_folder("inter_patient", "tissue_based", f"features_hunt{hunt}", f"tissues_{nb_tissues}")


def compute_tissue_absdif(cross_sect_data, id_target, id_candidate, seg_indexs):
    # NOTE: always uses the HUNT3 volumes, also when computing the HUNT4 similarity.
    fractions = np.zeros(len(seg_indexs))
    for i, seg_index in enumerate(seg_indexs):
        tissue_vol_target = cross_sect_data.loc[int(id_target)][f"{seg_index}_vol_hunt3"]
        tissue_vol_candidate = cross_sect_data.loc[int(id_candidate)][f"{seg_index}_vol_hunt3"]
        fractions[i] = abs(tissue_vol_candidate-tissue_vol_target)
    return fractions


def create_similarity_matrix(nb_tissues, tissues_ids=None, excluded_tissues=False, hunt=3):
    """Write ``<id>.csv`` with the similarity of ``id`` against every later patient (upper triangle).

    Args:
        tissues_ids: tissues to use (or to exclude with ``excluded_tissues=True``);
            default all tissues of the first patient's segmentation.
    """
    matchin_ids = splits.get_matching_ids()
    nb_ids = len(matchin_ids)

    # precomputed tissue volumes
    cross_sect_data = cross_sectional_features.FeaturesLoader(nb_tissues, matchin_ids).get_df()
    cross_sect_data.set_index('patient_id', inplace=True)

    seg_target = load_segmentation(matchin_ids[0], hunt, nb_tissues, remove_cerebellum=(nb_tissues == 3))
    used_tissues = np.unique(seg_target).astype(np.uint).tolist()
    if tissues_ids is not None:
        if excluded_tissues:
            used_tissues = [e for e in used_tissues if e not in tissues_ids]
        else:
            used_tissues = tissues_ids

    dice_columns = tuple(f'dice_{e}' for e in used_tissues)
    fraction_columns = tuple(f'absdif_{e}' for e in used_tissues)
    columns = ('id', 'dice_min', 'dice_max', 'dice_mean', 'absdif_min', 'absdif_max', 'absdif_mean') + dice_columns + fraction_columns

    nb_columns = len(columns)
    for index_target, id_target in enumerate(matchin_ids[:-1]):
        print(f"Processing {id_target} ({index_target+1}/{nb_ids-1})")

        df_path_name = f'{get_features_path(nb_tissues, hunt)}/{id_target}.csv'
        if io.verify_file(df_path_name):
            continue

        # the pairwise segmentations keep the cerebellum (only the tissue list above ignores it)
        seg_target = load_segmentation(id_target, hunt, nb_tissues)

        similarity_matrix = np.zeros((nb_ids-index_target-1, nb_columns))
        for c, id_candidate in enumerate(tqdm(matchin_ids[index_target+1:])):
            seg_candidate = load_segmentation(id_candidate, hunt, nb_tissues)

            dices_seg = metrics.compute_multiple_dice_coefficients(seg_target, seg_candidate, return_dict=False, seg_indexs=used_tissues)
            fractions_seg = compute_tissue_absdif(cross_sect_data, id_target, id_candidate, used_tissues)

            dices_seg_statistics = np.array([dices_seg.min(), dices_seg.max(), dices_seg.mean()])
            fractions_seg_statistics = np.array([fractions_seg.min(), fractions_seg.max(), fractions_seg.mean()])

            similarity_matrix[c] = np.concatenate((np.array([id_candidate]), dices_seg_statistics, fractions_seg_statistics, dices_seg, fractions_seg))

        save_table(pd.DataFrame(similarity_matrix, columns=columns), df_path_name)


class FeaturesLoader(PairwiseSimilarityLoader):
    def __init__(self, nb_tissues, ids, hunt=3):
        self.nb_tissues = nb_tissues
        super().__init__(get_features_path(nb_tissues, hunt), ids)
