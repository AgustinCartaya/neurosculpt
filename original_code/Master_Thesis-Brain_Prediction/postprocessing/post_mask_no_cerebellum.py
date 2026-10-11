
import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../'))
sys.path.append(base_path)

import numpy as np
import project_images_loader as pil
import utils.nifti_functions as nfc

import utils.util_freesurfer_segmentation as ufs
from scipy import ndimage


def get_circular_se(seg_2):
    radio = seg_2 // 2
    elemento = np.zeros((seg_2, seg_2, seg_2))
    centro = seg_2 // 2
    x, y, z = np.ogrid[:seg_2, :seg_2, :seg_2]
    distancia = np.sqrt((x - centro)**2 + (y - centro)**2 + (z - centro)**2)
    elemento[np.where(distancia <= radio)] = 1

    return elemento

def create_mask_no_cerebellum(bmask_path_name, seg_96_path_name, output_path_name):
    # 1: load bmask and segmentation 36
    bmask, affine = nfc.load_nifti(bmask_path_name)
    seg_96, _ = nfc.load_nifti(seg_96_path_name)
    seg_36 = ufs.merge_seg96_to_seg36(seg_96)
    
    # 2: remove cerebellum from segmentation 36 and obtain cerebellum mask
    seg_not_cerebellum, mask_cerebellum = ufs.remove_cerebellum(seg_36, seg_36, deep_copy=False, return_cerebellum_mask=True)

    # 3: create mask not touch (upper part of the brain)
    mask_not_touch = np.zeros(seg_36.shape)
    mask_not_touch[seg_not_cerebellum > 1] = 1

    # 4: dilate mask not touch to not lose cortex CSF
    se_1 = 7
    SE_1 = get_circular_se(se_1)
    mask_not_touch = ndimage.binary_dilation(mask_not_touch, structure=SE_1).astype(mask_not_touch.dtype)
    mask_not_touch = ndimage.binary_dilation(mask_not_touch, structure=SE_1).astype(mask_not_touch.dtype)
    mask_not_touch[mask_cerebellum == 1] = 0

    # 5: dilate mask cerebellum to remove cerebellum CSF
    se_2 = 15
    SE_2 = get_circular_se(se_2)
    mask_cerebellum = ndimage.binary_dilation(mask_cerebellum, structure=SE_2).astype(mask_cerebellum.dtype)
    mask_cerebellum = ndimage.binary_dilation(mask_cerebellum, structure=SE_2).astype(mask_cerebellum.dtype)
    mask_cerebellum[mask_not_touch==1] = 0

    # 6: create final mask
    bmask[mask_cerebellum==1] = 0
    bmask[seg_not_cerebellum == 1] = 1

    nfc.save_nifti(bmask.astype(np.uint8), affine, output_path_name)

