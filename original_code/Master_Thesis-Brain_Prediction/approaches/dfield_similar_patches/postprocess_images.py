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
import postprocessing.post_skull_stripping as post_skull_stripping
import postprocessing.post_normalization as post_normalization
import postprocessing.post_mask_no_cerebellum as post_mask_no_cerebellum

import project_images_loader as pil
import shutil


import constants as cts


def compute_statistics(ids, nb_images, nb_tissues):
    for i, id in enumerate(tqdm(ids)):
        if fc.verify_folder(cts.get_result_patient_folder(id, postprocessed=True), create=True):

            # # step 1: copy best raw image (fastsurfer pc)
            # img_res_raw_path_name = cts.get_result_image_path_name(id, "T1", (nb_tissues, nb_images))
            # img_res_raw_copy_path_name = cts.get_result_image_path_name(id, "T1_RAW", (nb_tissues, nb_images), postprocessed=True)
            # shutil.copy(img_res_raw_path_name, img_res_raw_copy_path_name)

            # # step 2: skull stripping (freesurfer pc)
            # img_res_raw_copy_path_name = cts.get_result_image_path_name(id, "T1_RAW", (nb_tissues, nb_images), postprocessed=True)
            # bmask_path_name = cts.get_result_image_path_name(id, "BMASK", (nb_tissues, nb_images), postprocessed=True)
            # post_skull_stripping.save_skull_stripping_freesurfer(img_res_raw_copy_path_name, bmask_path_name)

            # # # step 3: normalize image (fastsurfer pc)
            # img_res_raw_copy_path_name = cts.get_result_image_path_name(id, "T1_RAW", (nb_tissues, nb_images), postprocessed=True)
            # bmask_path_name = cts.get_result_image_path_name(id, "BMASK", (nb_tissues, nb_images), postprocessed=True)
            # img_post_path_name = cts.get_result_image_path_name(id, "T1_POST", (nb_tissues, nb_images), postprocessed=True)
            # post_normalization.save_normalized_image(img_res_raw_copy_path_name, bmask_path_name, img_post_path_name)

            # # # step 3: fastsurfer segmentation (fastsurfer pc)
            # img_post_path_name = cts.get_result_image_path_name(id, "T1_POST", (nb_tissues, nb_images), postprocessed=True)
            # seg_post_path_name = cts.get_result_image_path_name(id, "SEG_96", (nb_tissues, nb_images), postprocessed=True)
            # post_segment.segment_freesurfer(id, img_post_path_name, seg_post_path_name, GPU=True)

            # # # step 3.2: semanetation back to the original space (freesurfer pc)
            # img_post_path_name = cts.get_result_image_path_name(id, "T1_POST", (nb_tissues, nb_images), postprocessed=True)
            # seg_post_path_name = cts.get_result_image_path_name(id, "SEG_96", (nb_tissues, nb_images), postprocessed=True)
            # post_segment.transform_segmentation_to_patient_space(id, seg_post_path_name, img_post_path_name)

            # # # step 4: mask no cerebellum (fastsurfer/freesurfer pc)
            bmask_path_name = cts.get_result_image_path_name(id, "BMASK", (nb_tissues, nb_images), postprocessed=True)
            seg_post_path_name = cts.get_result_image_path_name(id, "SEG_96", (nb_tissues, nb_images), postprocessed=True)
            bmask_ncrb_path_name = cts.get_result_image_path_name(id, "BMASK_NCRB", (nb_tissues, nb_images), postprocessed=True)
            post_mask_no_cerebellum.create_mask_no_cerebellum(bmask_path_name, seg_post_path_name, bmask_ncrb_path_name)


val_images = ["08202"] + pil.get_matching_ids()[621:]
compute_statistics(val_images, 10, 96)