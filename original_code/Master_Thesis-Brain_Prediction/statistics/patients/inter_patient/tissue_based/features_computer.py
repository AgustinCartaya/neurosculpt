import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../../../'))

sys.path.append(base_path)


import numpy as np
import pandas as pd
import project_images_loader as pil
import utils.util as util
import utils.util_freesurfer_segmentation as ufs
import utils.nifti_functions as nfc
import utils.functions as fc
from tqdm import tqdm

cross_sectional_features = fc.load_module_from_path('cross_sectional_features', os.path.join(base_path, 'statistics/patients/cross_sectional/tissue_based/features_loader.py'))

def load_segmentation(id, hunt, nb_tissues, remove_nir=False):
    seg, _ = nfc.load_nifti(pil.get_segmentation_path_name(hunt, id, nb_tissues, absolute=True))
    if remove_nir:
        seg_36, _ = nfc.load_nifti(pil.get_segmentation_path_name(hunt, id, 36, absolute=True))
        return ufs.remove_cerebellum(seg, seg_36, deep_copy=False)
    return seg

# def load_tissue_volume():



def compute_multiple_dice_coefficients(masks1, masks2, seg_indexs=None):
    if seg_indexs is None:
        seg_indexs = np.unique(np.concatenate([masks1, masks2]).astype(np.uint))
    num_segments = len(seg_indexs)
    dices = np.zeros(num_segments)

    for i, seg_index in enumerate(seg_indexs):
        mask1 = masks1 == seg_index
        mask2 = masks2 == seg_index
        dices[i] = util.compute_dice_coefficient(mask1, mask2)
    return dices


def compute_tissue_absdif(cross_sect_data, id_target, id_candidate, seg_indexs):
    fractions = np.zeros(len(seg_indexs))
    for i, seg_index in enumerate(seg_indexs):
        tissue_vol_target = cross_sect_data.loc[int(id_target)][f"{seg_index}_vol_hunt3"]
        tissue_vol_candidate = cross_sect_data.loc[int(id_candidate)][f"{seg_index}_vol_hunt3"]
        fractions[i] = abs(tissue_vol_candidate-tissue_vol_target)
    return fractions


# def load_previous_dices(data_previous_dices, id_candidate, nb_tissues, prefix, dice_columns):
#     return 

def create_similarity_matrix(nb_tissues, tissues_ids=None, excluded_tissues=False, use_previous_dices=False, hunt=3):
    matchin_ids = pil.get_matching_ids()
    nb_ids = len(matchin_ids)

    # precomputed tissue volumens
    cross_sect_feat_loader = cross_sectional_features.FeaturesLoader(nb_tissues, matchin_ids)
    cross_sect_data = cross_sect_feat_loader.get_df()
    cross_sect_data.set_index('patient_id', inplace=True)

    # mask_target, _ = nfc.load_nifti(pil.get_segmentation_path_name(3, matchin_ids[0], nb_tissues=nb_tissues, absolute=True))
    seg_target = load_segmentation(matchin_ids[0], hunt, nb_tissues, remove_nir=(nb_tissues==3))
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

        if use_previous_dices:
            data_previous_dices = pd.read_csv(f"{current_path}/features/tissues_{nb_tissues}_only_dices/{id_target}.csv")
            data_previous_dices.set_index('id', inplace=True)

        df_path_name = f'{current_path}/features_hunt{hunt}/tissues_{nb_tissues}/{id_target}.csv'
        if fc.verify_file(df_path_name):
            continue

        seg_target = load_segmentation(id_target, hunt, nb_tissues)

        similarity_matrix = np.zeros((nb_ids-index_target-1, nb_columns))
        c=0
        for id_candidate in tqdm(matchin_ids[index_target+1:]):
            seg_candidate = load_segmentation(id_candidate, hunt, nb_tissues)

            if use_previous_dices:
                dices_seg = data_previous_dices.loc[int(id_candidate)][list(dice_columns)].values
            else:
                dices_seg = compute_multiple_dice_coefficients(seg_target, seg_candidate, used_tissues)
            fractions_seg = compute_tissue_absdif(cross_sect_data, id_target, id_candidate, used_tissues)

            dices_seg_statistics = np.array([dices_seg.min(), dices_seg.max(), dices_seg.mean()])
            fractions_seg_statistics = np.array([fractions_seg.min(), fractions_seg.max(), fractions_seg.mean()])
        
            similarity_matrix[c] = np.concatenate((np.array([id_candidate]), dices_seg_statistics, fractions_seg_statistics, dices_seg, fractions_seg))
            c+=1
            
            
        df_similarity = pd.DataFrame(similarity_matrix, columns=columns)
        df_similarity.to_csv(df_path_name, index=False)
        
        
# create_similarity_matrix(96, [0], excluded_tissues=True)
create_similarity_matrix(3, tissues_ids=[0], excluded_tissues=True, use_previous_dices=False, hunt=4)
































# def load_img_mask_and_skull_stripping(id, hunt):
#     img_target_path_name = pil.get_registered_to_template_image_path_name(hunt, id, pil.REGISTRATION_TO_MNI, absolute=True)
#     mask_target_path_name = pil.get_segmentation_path_name(hunt, id, simple=False, absolute=True)
    
#     img_target, _ = nfc.load_nifti(img_target_path_name)
#     mask_target, _ = nfc.load_nifti(mask_target_path_name)

#     img_target = util.skull_stripping_and_normalize(img_target, mask_target) 

#     return img_target, mask_target


# def get_metrics_dict(img_f, img_m):
#     rms = util.compute_RMS(img_f, img_m)
#     ncc = util.compute_Normalized_Cross_Correlation(img_f, img_m)
#     ssim = util.compute_ssim(img_f, img_m)

#     return {'rms': rms, 'ncc': ncc, 'ssim': ssim}


# def compute_multiple_dice_coefficients(masks1, masks2):
#     seg_indexs = np.unique(np.concatenate([masks1, masks2]).astype(np.uint))
#     num_segments = len(seg_indexs)
#     dices = np.zeros(num_segments)

#     for i, seg_index in enumerate(seg_indexs):
#         mask1 = masks1 == seg_index
#         mask2 = masks2 == seg_index
#         dices[i] = util.compute_dice_coefficient(mask1, mask2)

#     dices_dict = dict(zip(seg_indexs, dices))
#     return dices_dict
#     return dices


# def create_similarity_matrix():
#     matchin_ids = pil.get_matching_ids()
#     nb_ids = len(matchin_ids)
#     for index_target, id_target in enumerate(matchin_ids[:-1]):

#         print(f"Processing {id_target} ({index_target+1}/{nb_ids-1})")

#         img_target, mask_target = load_img_mask_and_skull_stripping(id_target, 3)

#         similarity_list = []

#         for id_candidate in tqdm(matchin_ids[index_target+1:index_target+2]):
#             img_candidate, mask_candidate = load_img_mask_and_skull_stripping(id_candidate, 3)

#             metrics_dict = {"id":id_candidate}

#             metrics_image = util.get_metrics_dict(img_target, img_candidate)
#             metrics_image = get_metrics_dict(img_target, img_candidate)
#             dices_seg = util.compute_multiple_dice_coefficients(mask_target, mask_candidate)
#             dices_seg_np = (pd.DataFrame([dices_seg])).to_numpy()
#             dices_seg_statistics = {"dice_min":dices_seg_np.min(), "dice_max":dices_seg_np.max(), "dice_mean":dices_seg_np.mean()}
            
#             metrics_dict.update(metrics_image)
#             metrics_dict.update(dices_seg_statistics)
#             metrics_dict.update(dices_seg)

#             similarity_list.append((metrics_dict))

#         df_metrics = pd.DataFrame(similarity_list)
#         df_metrics.to_csv(f'{current_path}/similarities/{id_target}.csv', index=False)

#     return df_metrics

# create_similarity_matrix()