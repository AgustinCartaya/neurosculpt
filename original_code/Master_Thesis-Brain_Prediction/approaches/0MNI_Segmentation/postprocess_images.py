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

def compute_statistics():
    # step 5: fastsurfer segmentation (fastsurfer pc)
    img_post_path_name = cts.get_result_image_path_name("T1")
    seg_post_path_name = cts.get_result_image_path_name("SEG_96")
    post_segment.segment_freesurfer("MNI", img_post_path_name, seg_post_path_name, GPU=True)

    # # # step 5.2: semanetation back to the original space (freesurfer pc)
    img_post_path_name = cts.get_result_image_path_name("T1")
    seg_post_path_name = cts.get_result_image_path_name("SEG_96")
    post_segment.transform_segmentation_to_patient_space(id, seg_post_path_name, img_post_path_name)

    # # # step 6: mask no cerebellum (fastsurfer/freesurfer pc)
    # bmask_path_name = cts.get_result_image_path_name(id, "BMASK", (ckeckpoint,), postprocessed=True)
    # seg_post_path_name = cts.get_result_image_path_name(id, "SEG_96_POST", (ckeckpoint,), postprocessed=True)
    # bmask_ncrb_path_name = cts.get_result_image_path_name(id, "BMASK_NCRB_POST", (ckeckpoint,), postprocessed=True)
    # post_mask_no_cerebellum.create_mask_no_cerebellum(bmask_path_name, seg_post_path_name, bmask_ncrb_path_name)



compute_statistics()

# 0.9664687684540453
# 0.9674776866480892