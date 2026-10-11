"""Step 7: brain mask without cerebellum (and without the CSF around it) -> ``BMASK_NCRB_MNI``."""
import numpy as np
from scipy import ndimage

from brainpred import paths
from brainpred.utils import io
from brainpred.utils import segmentation as ufs


def get_circular_se(size):
    """Spherical structuring element of diameter ``size``."""
    radius = size // 2
    element = np.zeros((size, size, size))
    center = size // 2
    x, y, z = np.ogrid[:size, :size, :size]
    distance = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
    element[np.where(distance <= radius)] = 1
    return element


def compute_mask_no_cerebellum(bmask, seg_36):
    """Remove the cerebellum and its surrounding CSF from ``bmask`` (modified in place) using ``seg_36``."""
    # 1: remove cerebellum from segmentation 36 and obtain cerebellum mask
    seg_not_cerebellum, mask_cerebellum = ufs.remove_cerebellum(seg_36, seg_36, deep_copy=False, return_cerebellum_mask=True)

    # 2: mask not to touch (upper part of the brain)
    mask_not_touch = np.zeros(seg_36.shape)
    mask_not_touch[seg_not_cerebellum > 1] = 1

    # 3: dilate mask not to touch, to keep the cortex CSF
    SE_1 = get_circular_se(7)
    mask_not_touch = ndimage.binary_dilation(mask_not_touch, structure=SE_1).astype(mask_not_touch.dtype)
    mask_not_touch = ndimage.binary_dilation(mask_not_touch, structure=SE_1).astype(mask_not_touch.dtype)
    mask_not_touch[mask_cerebellum == 1] = 0

    # 4: dilate cerebellum mask, to remove the cerebellum CSF
    SE_2 = get_circular_se(15)
    mask_cerebellum = ndimage.binary_dilation(mask_cerebellum, structure=SE_2).astype(mask_cerebellum.dtype)
    mask_cerebellum = ndimage.binary_dilation(mask_cerebellum, structure=SE_2).astype(mask_cerebellum.dtype)
    mask_cerebellum[mask_not_touch == 1] = 0

    # 5: final mask
    bmask[mask_cerebellum == 1] = 0
    bmask[seg_not_cerebellum == 1] = 1
    return bmask.astype(np.uint8)


def create_mask_no_cerebellum(hunt, id):
    bmask, affine = io.load_nifti(paths.get_brain_mask_path_name(hunt, id, paths.REGISTRATION_TO_MNI))
    seg_36, _ = io.load_nifti(paths.get_segmentation_path_name(hunt, id, 36))
    io.save_nifti(compute_mask_no_cerebellum(bmask, seg_36), affine,
                  paths.get_brain_mask_path_name(hunt, id, paths.REGISTRATION_TO_MNI, no_cerebellum=True))


def create_mask_no_cerebellum_from_seg_96(bmask_path_name, seg_96_path_name, output_path_name):
    """Same as :func:`create_mask_no_cerebellum` for arbitrary files, starting from a 96-label segmentation."""
    bmask, affine = io.load_nifti(bmask_path_name)
    seg_96, _ = io.load_nifti(seg_96_path_name)
    seg_36 = ufs.merge_seg96_to_seg36(seg_96)
    io.save_nifti(compute_mask_no_cerebellum(bmask, seg_36), affine, output_path_name)
