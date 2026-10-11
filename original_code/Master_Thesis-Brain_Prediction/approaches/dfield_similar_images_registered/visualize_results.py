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


import project_images_loader as pil
import utils.util as util
import utils.functions as fc
import utils.nifti_functions as nfc
import utils.util_freesurfer_segmentation as ufs

import constants as cts

val_images = ["08202"] + pil.get_matching_ids()[621:]

def visualize_results(id, nb_images_list, reg_type_list):
    nb_show_images =  len(reg_type_list) * len(nb_images_list) + 2
    img_list = [None] *nb_show_images
    name_list = [None] *nb_show_images

    # load original image 
    img_list[0] = nfc.load_nifti(pil.get_registered_to_template_image_path_name(3, id, pil.REGISTRATION_TO_MNI, absolute=True))[0]
    img_list[1] = nfc.load_nifti(pil.get_registered_to_template_image_path_name(4, id, pil.REGISTRATION_TO_MNI, absolute=True))[0]
    name_list[:2] = [f"HUNT3", f"HUNT4"]

    # lad results
    index = 0
    for j, reg_type in enumerate(reg_type_list):
        for k, nb_images in enumerate(nb_images_list):
            img_list[index+2] = nfc.load_nifti(cts.get_result_image_path_name(id, "T1", (reg_type, nb_images)))[0]
            name_list[index+2] = f"{reg_type}_{nb_images}"
            index += 1

    fc.imgshow_3D_list(f"Results dfield Similar images registered{id}", img_list, subimg_titles=name_list, mooving_plane=2, custom_disposition=(2,-1), subtitle_help=(20,5), subtitle_space=20)


def visualize_results_diff(id, nb_images_list, reg_type_list):
    nb_show_images =  len(reg_type_list) * len(nb_images_list) + 2
    img_list = [None] *nb_show_images
    name_list = [None] *nb_show_images

    # load original image 
    img_org_hunt4 = nfc.load_nifti(pil.get_registered_to_template_image_path_name(4, id, pil.REGISTRATION_TO_MNI, absolute=True))[0]
    seg_org_hunt4 = nfc.load_nifti(pil.get_segmentation_path_name(4, id, 96, absolute=True))[0]
    img_org_hunt4 = util.skull_stripping_and_normalize(img_org_hunt4, seg_org_hunt4)

    img_list[0] = np.abs(img_org_hunt4 - img_org_hunt4)
    img_org_hunt3 = nfc.load_nifti(pil.get_registered_to_template_image_path_name(3, id, pil.REGISTRATION_TO_MNI, absolute=True))[0]
    seg_org_hunt3 = nfc.load_nifti(pil.get_segmentation_path_name(3, id, 96, absolute=True))[0]
    img_org_hunt3 = util.skull_stripping_and_normalize(img_org_hunt3, seg_org_hunt3)

    # img_list[1] = standardize_array(img_org_hunt3 - img_org_hunt4)
    img_list[1] = np.abs(img_org_hunt3 - img_org_hunt4)
    name_list[:2] = [f"HUNT4", f"HUNT3"]

    # lad results
    index = 0
    for j, reg_type in enumerate(reg_type_list):
        for k, nb_images in enumerate(nb_images_list):
            img_res, _ = nfc.load_nifti(cts.get_result_image_path_name(id, "T1", (reg_type, nb_images,)))
            seg_res, _ = nfc.load_nifti(cts.get_result_image_path_name(id, "SEG_96", (reg_type, nb_images,)))
            img_res = util.skull_stripping_and_normalize(img_res, seg_res)
        
            img_list[index+2] = np.abs(img_res - img_org_hunt4)
            name_list[index+2] = f"{reg_type}_{nb_images}"
            index += 1
    fc.imgshow_3D_list(f"Results dfield Similar images registered {id}", img_list, subimg_titles=name_list, mooving_plane=2, custom_disposition=(2,-1), subtitle_help=(20,5), subtitle_space=20, cmap="hot")



def visualize_results_and_seg(id, nb_images_list, reg_type_list):
    nb_show_images =  len(reg_type_list) * len(nb_images_list) + 1
    img_list = [None] *nb_show_images*2
    name_list = [""] *nb_show_images*2

    # load original image 
    img_org_hunt4 = nfc.load_nifti(pil.get_registered_to_template_image_path_name(4, id, pil.REGISTRATION_TO_MNI, absolute=True))[0]
    seg_org_hunt4 = nfc.load_nifti(pil.get_segmentation_path_name(4, id, 96, absolute=True))[0]
    img_org_hunt4 = util.skull_stripping_and_normalize(img_org_hunt4, seg_org_hunt4)

    img_list[0] = np.stack([img_org_hunt4, img_org_hunt4, img_org_hunt4], axis=3)
    img_list[nb_show_images] = ufs.seg_to_rgbseg(seg_org_hunt4)/255
    name_list[0] = f"HUNT4"    

    # lad results

    index = 0
    progress_bar = tqdm(total=nb_show_images-1)
    for j, reg_type in enumerate(reg_type_list):
        for k, nb_images in enumerate(nb_images_list):
            img_res, _ = nfc.load_nifti(cts.get_result_image_path_name(id, "T1", (reg_type, nb_images,)))
            seg_res, _ = nfc.load_nifti(cts.get_result_image_path_name(id, "SEG_96", (reg_type, nb_images,)))
            img_res = util.skull_stripping_and_normalize(img_res, seg_res)

            img_list[index+1] =  np.stack([img_res, img_res, img_res], axis=3)
            img_list[index+1+nb_show_images] =  ufs.seg_to_rgbseg(seg_res)/255

            name_list[index+1] = f"{reg_type}_{nb_images}"
            progress_bar.update(1)
            index += 1


    fc.imgshow_3D_list(f"Results dfield Similar images registered {id}", img_list, subimg_titles=name_list, mooving_plane=2, custom_disposition=(2,-1), subtitle_help=(20,5), subtitle_space=20)



# visualize_results(val_images[0], [5], [cts.PARAMETERS_FILE_AFFINE, cts.PARAMETERS_FILE_BSPLINE_FAST])
# visualize_results_diff(val_images[0], [5], [cts.PARAMETERS_FILE_AFFINE, cts.PARAMETERS_FILE_BSPLINE_FAST])
visualize_results_and_seg(val_images[0], [5], [cts.PARAMETERS_FILE_AFFINE, cts.PARAMETERS_FILE_BSPLINE_FAST])

