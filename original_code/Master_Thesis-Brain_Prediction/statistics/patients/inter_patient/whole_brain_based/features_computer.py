import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../../../'))

sys.path.append(base_path)

import numpy as np
import pandas as pd
import project_images_loader as pil
import utils.util as util
import utils.nifti_functions as nfc
import utils.functions as fc
from tqdm import tqdm


def load_img_mask_and_skull_stripping(id, hunt, nb_tissues=3):
    img_target_path_name = pil.get_registered_to_template_image_path_name(hunt, id, pil.REGISTRATION_TO_MNI, absolute=True)
    mask_target_path_name = pil.get_segmentation_path_name(hunt, id, nb_tissues, absolute=True)
    
    img_target, _ = nfc.load_nifti(img_target_path_name)
    mask_target, _ = nfc.load_nifti(mask_target_path_name)

    img_target = util.skull_stripping_and_normalize(img_target, mask_target) 
    return img_target


def get_img_sim_metrics(img_f, img_m):
    ncc = util.compute_normalized_cross_correlation(img_f, img_m)
    ssim = util.compute_ssim(img_f, img_m)
    return [ncc, ssim]


def create_similarity_matrix():
    matchin_ids = pil.get_matching_ids()
    nb_ids = len(matchin_ids)

    columns = ['id', 'ncc', 'ssim']
    nb_columns = len(columns)
    for index_target, id_target in enumerate(matchin_ids[:-1]):

        print(f"Processing {id_target} ({index_target+1}/{nb_ids-1})")

        df_path_name = f'{current_path}/features/{id_target}.csv'
        if fc.verify_file(df_path_name):
            continue

        hunt = 4
        img_target = load_img_mask_and_skull_stripping(id_target, hunt)
        similarity_matrix = np.zeros((nb_ids-index_target-1, nb_columns))
        for i, id_candidate in enumerate(tqdm(matchin_ids[index_target+1:])):
            img_candidate = load_img_mask_and_skull_stripping(id_candidate, hunt)

            metrics_image = np.array(get_img_sim_metrics(img_target, img_candidate))
            similarity_matrix[i] = np.concatenate((np.array([id_candidate]), metrics_image))

        df_similarity = pd.DataFrame(similarity_matrix, columns=columns)
        df_similarity.to_csv(df_path_name, index=False)

create_similarity_matrix()






























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