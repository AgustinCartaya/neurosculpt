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

def load_original_img(id, hunt, skull_stripping=True):
    img = nfc.load_nifti(pil.get_registered_to_template_image_path_name(hunt, id, pil.REGISTRATION_TO_MNI, absolute=True))[0]
    seg = nfc.load_nifti(pil.get_segmentation_path_name(hunt, id, 96, absolute=True))[0]
    if skull_stripping:
        img = util.skull_stripping_and_normalize(img, seg)
    else:
        img = util.normalize(img, seg)
    return img, seg

def load_results_img(id, checkpoint, postprocessed=False, force_raw=True, skull_stripping=False):
    if postprocessed:
        suffix = "_RAW" if force_raw else "_POST"
    else:
        suffix =  ""

    img = nfc.load_nifti(cts.get_result_image_path_name(id, "T1"+suffix, (checkpoint,), postprocessed=postprocessed))[0]
    seg = None
    if skull_stripping:
        suffix = "_RAW" if force_raw else "_POST"
        seg = nfc.load_nifti(cts.get_result_image_path_name(id, "SEG_96" + suffix, (52,), postprocessed=True))[0]
        img = util.skull_stripping_and_normalize(img, seg)
    return img, seg


def visualize_results(id, checkpoints_list, postprocessed=False, force_raw=True, skull_stripping=True):
    nb_show_images = len(checkpoints_list) + 2
    img_list = [None] *nb_show_images
    name_list = [None] *nb_show_images

    # load original image 
    img_list[0] = load_original_img(id, 3, skull_stripping)[0]
    img_list[1] = load_original_img(id, 4, skull_stripping)[0]
    name_list[:2] = [f"HUNT3", f"HUNT4"]

    # lad results
    index = 0
    for j, checkpoint in enumerate(checkpoints_list):
        img_list[index+2] = load_results_img(id, checkpoint, postprocessed, force_raw, skull_stripping)[0]
        name_list[index+2] = f"{checkpoint}"
        index += 1
    custom_disposition = (2, -1) if nb_show_images > 4 else (1, -1)
    fc.imgshow_3D_list(f"Results MPGAN Seg loss {id}", img_list, subimg_titles=name_list, mooving_plane=2, custom_disposition=custom_disposition, subtitle_help=(20,5), subtitle_space=20)


def visualize_results_diff(id, checkpoints_list, postprocessed=False, force_raw=True):
    nb_show_images = len(checkpoints_list) + 2
    img_list = [None] *nb_show_images
    name_list = [None] *nb_show_images

    # load original image 
    img_org_hunt4 = load_original_img(id, 4, skull_stripping=True)[0]
    img_org_hunt3 = load_original_img(id, 3, skull_stripping=True)[0]
    img_list[0] = np.abs(img_org_hunt4 - img_org_hunt4)
    img_list[0] = np.ones_like(img_org_hunt4) # just to make the hot colormap work with the range [0,1]
    img_list[1] = np.abs(img_org_hunt3 - img_org_hunt4)
    name_list[:2] = [f"HUNT4", f"HUNT3"]

    # lad results
    index = 0
    for j, checkpoint in enumerate(checkpoints_list):
            img_res = load_results_img(id, checkpoint, postprocessed=postprocessed, force_raw=force_raw, skull_stripping=True)[0]

            img_list[index+2] = np.abs(img_res - img_org_hunt4)
            name_list[index+2] = f"{checkpoint}"
            index += 1
    custom_disposition = (2, -1) if nb_show_images > 4 else (1, -1)
    fc.imgshow_3D_list(f"Results MPGAN Seg loss {id}", img_list, subimg_titles=name_list, mooving_plane=2, custom_disposition=custom_disposition, subtitle_help=(20,5), subtitle_space=20, cmap="hot")



def visualize_results_and_seg(id, checkpoints_list, force_raw=True):
    nb_show_images = len(checkpoints_list) + 1
    img_list = [None] *nb_show_images*2
    name_list = [""] *nb_show_images*2

    # load original image 
    img_org_hunt4, seg_org_hunt4  = load_original_img(id, 4, skull_stripping=True)

    img_list[0] = np.stack([img_org_hunt4, img_org_hunt4, img_org_hunt4], axis=3)
    img_list[nb_show_images] = ufs.seg_to_rgbseg(seg_org_hunt4)/255
    name_list[0] = f"HUNT4"    

    # lad results

    progress_bar = tqdm(total=nb_show_images-1)
    index = 0
    for j, checkpoint in enumerate(checkpoints_list):
        img_res, seg_res = load_results_img(id, checkpoint, postprocessed=True, force_raw=force_raw, skull_stripping=True)

        img_list[index+1] =  np.stack([img_res, img_res, img_res], axis=3)
        img_list[index+1+nb_show_images] =  ufs.seg_to_rgbseg(seg_res)/255

        name_list[index+1] = f"{checkpoint}"
        progress_bar.update(1)
        index += 1

    fc.imgshow_3D_list(f"Results MPGAN Seg loss {id}", img_list, subimg_titles=name_list, mooving_plane=2, custom_disposition=(2,-1), subtitle_help=(20,5), subtitle_space=20)

# visualize_results(val_images[0], [1, 52, 76], postprocessed=False, skull_stripping=False)
visualize_results(val_images[13], [52], postprocessed=True, skull_stripping=True, force_raw=False)
# visualize_results_diff(val_images[0], [1, 52, 76], postprocessed=False)
# visualize_results_diff(val_images[0], [52], postprocessed=True, force_raw=True)
# visualize_results_and_seg(val_images[0], [52], force_raw=True)

