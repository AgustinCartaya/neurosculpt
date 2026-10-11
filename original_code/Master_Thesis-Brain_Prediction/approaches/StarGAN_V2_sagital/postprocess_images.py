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
import test_code.creator_2D_images.constants as cts_prep_mni


# import itk
# def save_image_back(id, img_res_raw_path_name, img_registered_path_name):
#     param_obj = itk.ParameterObject.New()
#     param_reg_path_name = cts_prep_mni.get_INV_registered_pfile_path_name(3, id)
    
#     param_obj.AddParameterFile(param_reg_path_name)

#     img_moving = itk.imread(img_res_raw_path_name, itk.F)
#     reg_img_moving = itk.transformix_filter(img_moving, param_obj)

#     itk.imwrite(reg_img_moving, img_registered_path_name)


def compute_statistics(ids, style="BEST"):
    for i, id in enumerate(tqdm(ids)):
        print("id", id)
        if fc.verify_folder(cts.get_result_patient_folder(id, postprocessed=True), create=True):
            # # # step 1: Register images back the prep space
            # img_res_raw_path_name = cts.get_result_image_path_name(id, "T1", (style, ), postprocessed=False, prep_mni=True)
            # img_reg_path_name = cts.get_result_image_path_name(id, "T1_REG", (style,), postprocessed=True, prep_mni=True)
            # save_image_back(id, img_res_raw_path_name, img_reg_path_name)

            # # # step 3: skull stripping (freesurfer pc)
            # img_reg_path_name = cts.get_result_image_path_name(id, "T1_REG", (style,), postprocessed=True, prep_mni=True)
            # bmask_path_name = cts.get_result_image_path_name(id, "BMASK", (style,), postprocessed=True)
            # post_skull_stripping.save_skull_stripping_freesurfer(img_reg_path_name, bmask_path_name)

            # # # # # step 4: normalize image (freesurfer pc)
            # img_reg_path_name = cts.get_result_image_path_name(id, "T1_REG", (style,), postprocessed=True, prep_mni=True)
            # bmask_path_name = cts.get_result_image_path_name(id, "BMASK", (style,), postprocessed=True)
            # img_post_path_name = cts.get_result_image_path_name(id, "T1_POST", (style,), postprocessed=True, prep_mni=True)
            # post_normalization.save_normalized_image(img_reg_path_name, bmask_path_name, img_post_path_name)

            # # step 5: fastsurfer segmentation (fastsurfer pc)
            # img_post_path_name = cts.get_result_image_path_name(id, "T1_POST", (style,), postprocessed=True, prep_mni=True)
            # seg_post_path_name = cts.get_result_image_path_name(id, "SEG_96_POST", (style,), postprocessed=True, prep_mni=True)
            # post_segment.segment_freesurfer(id, img_post_path_name, seg_post_path_name, GPU=True)

            # # # # step 5.2: semanetation back to the original space (freesurfer pc)
            # img_post_path_name = cts.get_result_image_path_name(id, "T1_POST", (style,), postprocessed=True, prep_mni=True)
            # seg_post_path_name = cts.get_result_image_path_name(id, "SEG_96_POST", (style,), postprocessed=True, prep_mni=True)
            # post_segment.transform_segmentation_to_patient_space(id, seg_post_path_name, img_post_path_name)
          
            # # # # step 6: mask no cerebellum (fastsurfer/freesurfer pc)
            bmask_path_name = cts.get_result_image_path_name(id, "BMASK", (style,), postprocessed=True)
            seg_post_path_name = cts.get_result_image_path_name(id, "SEG_96_POST", (style,), postprocessed=True, prep_mni=True)
            bmask_ncrb_path_name = cts.get_result_image_path_name(id, "BMASK_NCRB_POST", (style,), postprocessed=True)
            post_mask_no_cerebellum.create_mask_no_cerebellum(bmask_path_name, seg_post_path_name, bmask_ncrb_path_name)
            
      


val_images = ["08202"] + pil.get_matching_ids()[621:]
compute_statistics(val_images)

# 0.9664687684540453
# 0.9674776866480892