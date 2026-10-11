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

import project_images_loader as pil
import shutil


import constants as cts


def compute_statistics(ids, nb_images, sim_metric):
    for i, id in enumerate(tqdm(ids)):
        if fc.verify_folder(cts.get_result_patient_folder(id, postprocessed=True), create=True):
            # postprocessing (None)
            # img_res_path_name = cts.get_result_image_path_name(id, "T1", (sim_metric, nb_images))
            # img_postprocesses_path_name_ = cts.get_result_image_path_name(id, "T1", (sim_metric, nb_images), postprocessed=True)
            # shutil.copy(img_res_path_name, img_postprocesses_path_name_)


            # segment image
            img_postprocesses_path_name_ = cts.get_result_image_path_name(id, "T1", (sim_metric, nb_images), postprocessed=True)
            seg_postprocessed_path_name = cts.get_result_image_path_name(id, "SEG_96", (sim_metric, nb_images), postprocessed=True)
            # step 1: segment freesurfer
            # post_segment.segment_freesurfer(id, img_postprocesses_path_name_, seg_postprocessed_path_name, GPU=True)

            # step 1.2: semanetation back to the original space
            post_segment.transform_segmentation_to_patient_space(id, seg_postprocessed_path_name, img_postprocesses_path_name_)
        if i == 15:
            break
        

val_images = ["08202"] + pil.get_matching_ids()[621:]
compute_statistics(val_images, 100, 'dice_mean'.upper())