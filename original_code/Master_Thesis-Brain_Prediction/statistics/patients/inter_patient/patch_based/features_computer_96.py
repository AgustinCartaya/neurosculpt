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
import utils.patches_controller as patches_controller




def compute_tissue_weigthed_dice(patch_target, patch_data):
    tissue_in_target, nb_tissue_voxels_in_target = np.unique(patch_target, return_counts=True)
    tissue_in_data, nb_tissue_voxels_in_data = np.unique(patch_target, return_counts=True)

    tissue_volume_in_target = nb_tissue_voxels_in_target / np.prod(patch_target.shape)
    tissue_volume_in_data = nb_tissue_voxels_in_data / np.prod(patch_data.shape)

    weighted_sum = 0
    total_tissue_weight = 0
    for i, tissue_index in enumerate(tissue_in_target):
        if tissue_index in tissue_in_data:
            tissue_weight_target = tissue_volume_in_target[i]

            i_data = np.where(tissue_in_data == tissue_index)[0][0]
            tissue_weight_data = tissue_volume_in_data[i_data]

            mask1 = patch_target == tissue_index
            mask2 = patch_data == tissue_index

            tissue_dice = util.compute_dice_coefficient(mask1, mask2)
            tissue_weight = 0
            weighted_sum += (tissue_weight + tissue_weight_target + tissue_weight_data) * tissue_dice
            total_tissue_weight += tissue_weight

    dice = weighted_sum / (2+total_tissue_weight)
    return dice


def compute_patches_similarity_table(seg_target, ids_dataset, nb_tissues, patch_shape=(32,32,32), extraction_step=(16,16,16)):
    nb_max_data_images = len(ids_dataset)
    
    # 1. extract patches
    patches_target = patches_controller.extract_patches(seg_target, patch_shape, extraction_step)
 
    # 2. compute dice for each patch and each image
    similarity_patches_table = np.zeros((nb_max_data_images, len(patches_target)+1))
    img_index = 0

    for id_data in tqdm(ids_dataset):
        mask_data, _ = nfc.load_nifti(pil.get_segmentation_path_name(3, id_data, nb_tissues=nb_tissues, absolute=True))
        patches_data = patches_controller.extract_patches(mask_data, patch_shape, extraction_step)
        similarity_patches_table[img_index, 0] = int(id_data)
        for patch_index, patch_target in enumerate(patches_target):
            if np.any(patch_target > 0) and np.any(patches_data[patch_index] > 0):
                similarity_patches_table[img_index, patch_index+1] = compute_tissue_weigthed_dice(patch_target, patches_data[patch_index])
            
        img_index += 1
    return similarity_patches_table


def create_and_save_similarity_patches_tables(ids_val, isd_train, nb_tissues):
    for id_val in ids_val:
        print(f"Processing {id_val} nb_tissues: {nb_tissues}")
        seg_target, _ = nfc.load_nifti(pil.get_segmentation_path_name(3, id_val, nb_tissues, absolute=True))
        similarity_patches_table = compute_patches_similarity_table(seg_target, isd_train, nb_tissues)
        # similarity_patches_table = np.concatenate([np.array([isd_train]).astype(int), similarity_patches_table], axis=1)

        columns = ["id"]+[f"patch_{i}" for i in range(similarity_patches_table.shape[1]-1)]
        df_patient = pd.DataFrame(similarity_patches_table, columns=columns)

        # patient_features_path = f'{current_path}/features/tissues_{nb_tissues}/{id_val}.csv'
        # if not fc.verify_file(patient_features_path):
        #     df_patient.to_csv(patient_features_path, index=False)



ids_val = ["08202"] +  pil.get_matching_ids()[621:]
ids_train = pil.get_matching_ids()[:621]
ids_train = [id for id in ids_train if id not in pil.EXCLUDED_PATIENTS + ["08202"]]

create_and_save_similarity_patches_tables(ids_val, ids_train, 96)



















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