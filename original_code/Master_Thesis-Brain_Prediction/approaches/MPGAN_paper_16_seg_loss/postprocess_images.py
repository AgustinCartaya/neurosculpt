import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))

sys.path.append(base_path)

import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
from tqdm import tqdm
import pandas as pd
from IPython.display import display

import utils.util as util
import utils.nifti_functions as nfc
import utils.functions as fc
import pandas as pd
import postprocessing.post_segment as post_segment
import postprocessing.post_synthsr as post_synthsr
import postprocessing.post_hist_matching as post_hist_matching
import postprocessing.post_skull_stripping as post_skull_stripping
import postprocessing.post_normalization as post_normalization
import postprocessing.post_mask_no_cerebellum as post_mask_no_cerebellum

import project_images_loader as pil
import shutil


import constants as cts
features_loader_tissue_based = fc.load_module_from_path('features_loader_tissue_based', os.path.join(base_path, 'statistics/patients/inter_patient/tissue_based/features_loader.py'))
dice_mean_table_reder = features_loader_tissue_based.FeaturesLoader(3, pil.get_matching_ids())


def compute_statistics(ids, ckeckpoint):
    for i, id in enumerate(tqdm(ids)):
        print("id", id)
        if fc.verify_folder(cts.get_result_patient_folder(id, postprocessed=True), create=True):
            # # step 1: copy images to apply synthsr
            # img_res_raw_path_name = cts.get_result_image_path_name(id, "T1", (ckeckpoint, ))
            # img_res_raw_copy_path_name = cts.get_result_image_path_name(id, "T1_RAW", (ckeckpoint,), postprocessed=True)
            # shutil.copy(img_res_raw_path_name, img_res_raw_copy_path_name)

            ### ------------------------ NOT WORKING ------------------------ ###
            # # # step 1.other: apply hitogram matching (is not better than raw image)
            # df_patient_sim_table = dice_mean_table_reder.get_sorted_similarity_table(id, "dice_mean", 1, exclude_ids= ids + pil.EXCLUDED_PATIENTS)
            # img_ref_path_name_path_name = pil.get_preprocessed_image_path_name(4, df_patient_sim_table.iloc[0,0], absolute=True)
            # img_org_path_name_path_name = pil.get_preprocessed_image_path_name(4, id, absolute=True)
            # img_res_raw_copy_path_name = cts.get_result_image_path_name(id, "T1_RAW", (ckeckpoint,), postprocessed=True)
            # img_res_hist_match_path_name = cts.get_result_image_path_name(id, "T1_HISTMATCH", (ckeckpoint,), postprocessed=True)
            # post_hist_matching.save_histogram_match(img_ref_path_name_path_name, img_res_raw_copy_path_name, img_res_hist_match_path_name)

            # img_org, _ = nfc.load_nifti(img_org_path_name_path_name)
            # img_res_raw, _ = nfc.load_nifti(img_res_raw_copy_path_name)
            # img_res_hist_match, _ = nfc.load_nifti(img_res_hist_match_path_name)
            # print(util.compute_ssim(img_org, img_res_raw))
            # print(util.compute_ssim(img_org, img_res_hist_match))
            ### ------------------------ NOT WORKING ------------------------ ###

            # # # step 2: apply synthsr
            # img_res_raw_copy_path_name = cts.get_result_image_path_name(id, "T1_RAW", (ckeckpoint,), postprocessed=True)
            # img_res_synthsr_path_name = cts.get_result_image_path_name(id, "T1_SYNTHSR", (ckeckpoint,), postprocessed=True)
            # post_synthsr.save_synthSR(img_res_raw_copy_path_name, img_res_synthsr_path_name)

            # # # step 3: skull stripping (freesurfer pc)
            # img_res_synthsr_path_name = cts.get_result_image_path_name(id, "T1_SYNTHSR", (ckeckpoint,), postprocessed=True)
            # bmask_path_name = cts.get_result_image_path_name(id, "BMASK", (ckeckpoint,), postprocessed=True)
            # post_skull_stripping.save_skull_stripping_freesurfer(img_res_synthsr_path_name, bmask_path_name)

            # # # # step 4: normalize image (freesurfer pc)
            # img_res_synthsr_path_name = cts.get_result_image_path_name(id, "T1_SYNTHSR", (ckeckpoint,), postprocessed=True)
            # bmask_path_name = cts.get_result_image_path_name(id, "BMASK", (ckeckpoint,), postprocessed=True)
            # img_post_path_name = cts.get_result_image_path_name(id, "T1_POST", (ckeckpoint,), postprocessed=True)
            # post_normalization.save_normalized_image(img_res_synthsr_path_name, bmask_path_name, img_post_path_name)

            ### ------------ STOP HERE AND COMPARE SSIM BETWEEN RAW AND POST ------------ ###
            # img_org_path_name_path_name = pil.get_preprocessed_image_path_name(4, id, absolute=True)
            # img_res_raw_copy_path_name = cts.get_result_image_path_name(id, "T1_RAW", (ckeckpoint,), postprocessed=True)
            # img_post_path_name = cts.get_result_image_path_name(id, "T1_POST", (ckeckpoint,), postprocessed=True)
            # img_post_match = post_hist_matching.save_histogram_match(img_res_raw_copy_path_name, img_post_path_name, None, just_return=True)
            # img_org, _ = nfc.load_nifti(img_org_path_name_path_name)
            # img_res_raw, _ = nfc.load_nifti(img_res_raw_copy_path_name)
            # # img_post, _ = nfc.load_nifti(img_post_path_name)
            # print(util.compute_ssim(img_org, img_res_raw))
            # print(util.compute_ssim(img_org, img_post_match))
            ### ------------ STOP HERE AND COMPARE SSIM BETWEEN RAW AND SYNTHSR ------------ ###
            # In the following steps 2 types of postprocessing (segmentations are performed)
            # 1: with the raw image (this is the best one so far)
            # 2: with the synthsr posrprocessed image



            # # # step 5: fastsurfer segmentation (fastsurfer pc)
            # img_post_path_name = cts.get_result_image_path_name(id, "T1_POST", (ckeckpoint,), postprocessed=True)
            # seg_post_path_name = cts.get_result_image_path_name(id, "SEG_96_POST", (ckeckpoint,), postprocessed=True)
            # post_segment.segment_freesurfer(id, img_post_path_name, seg_post_path_name, GPU=True)

            # # # step 5.1: fastsurfer segmentation (fastsurfer pc)
            # img_res_raw_copy_path_name = cts.get_result_image_path_name(id, "T1_RAW", (ckeckpoint,), postprocessed=True)
            # seg_raw_path_name = cts.get_result_image_path_name(id, "SEG_96_RAW", (ckeckpoint,), postprocessed=True)
            # post_segment.segment_freesurfer(id, img_res_raw_copy_path_name, seg_raw_path_name, GPU=True)

            # # # step 5.2: semanetation back to the original space (freesurfer pc)
            # img_post_path_name = cts.get_result_image_path_name(id, "T1_POST", (ckeckpoint,), postprocessed=True)
            # seg_post_path_name = cts.get_result_image_path_name(id, "SEG_96_POST", (ckeckpoint,), postprocessed=True)
            # post_segment.transform_segmentation_to_patient_space(id, seg_post_path_name, img_post_path_name)

            # # # step 5.3: semanetation back to the original space (freesurfer pc)
            # img_res_raw_copy_path_name = cts.get_result_image_path_name(id, "T1_RAW", (ckeckpoint,), postprocessed=True)
            # seg_raw_path_name = cts.get_result_image_path_name(id, "SEG_96_RAW", (ckeckpoint,), postprocessed=True)
            # post_segment.transform_segmentation_to_patient_space(id, seg_raw_path_name, img_res_raw_copy_path_name)

            # # # step 6: mask no cerebellum (fastsurfer/freesurfer pc)
            bmask_path_name = cts.get_result_image_path_name(id, "BMASK", (ckeckpoint,), postprocessed=True)
            seg_post_path_name = cts.get_result_image_path_name(id, "SEG_96_POST", (ckeckpoint,), postprocessed=True)
            bmask_ncrb_path_name = cts.get_result_image_path_name(id, "BMASK_NCRB_POST", (ckeckpoint,), postprocessed=True)
            post_mask_no_cerebellum.create_mask_no_cerebellum(bmask_path_name, seg_post_path_name, bmask_ncrb_path_name)
            
            # # # step 6.1: mask no cerebellum (fastsurfer/freesurfer pc)
            bmask_path_name = cts.get_result_image_path_name(id, "BMASK", (ckeckpoint,), postprocessed=True)
            seg_post_path_name = cts.get_result_image_path_name(id, "SEG_96_RAW", (ckeckpoint,), postprocessed=True)
            bmask_ncrb_path_name = cts.get_result_image_path_name(id, "BMASK_NCRB_RAW", (ckeckpoint,), postprocessed=True)
            post_mask_no_cerebellum.create_mask_no_cerebellum(bmask_path_name, seg_post_path_name, bmask_ncrb_path_name)



val_images = ["08202"] + pil.get_matching_ids()[621:]
compute_statistics(val_images, ckeckpoint=52)

# 0.9664687684540453
# 0.9674776866480892