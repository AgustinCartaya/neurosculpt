"""Apply a predicted deformation field to the HUNT3 image, segmentation and brain mask of each validation patient."""
import os

import numpy as np
from tqdm import tqdm

from brainpred import paths
from brainpred.utils import io
from brainpred.utils.dfield import apply_deformation_field_elastix

# result type -> (HUNT3 image type to load, its extension, binary interpolation)
IMAGE_TYPES = {
    "T1": ("T1_MNI", "nii", False),
    "SEG_96": ("SEG_96", "nii.gz", True),
    "BMASK_NCRB": ("BMASK_NCRB_MNI", "nii.gz", True),
}


def register_image(img_path_name, dfield_path_name, binary=False):
    img_reg = apply_deformation_field_elastix(img_path_name, dfield_path_name, binary=binary)
    if binary:
        img_reg = img_reg.astype(np.uint16)
    return img_reg


def save_result(img_reg, result_path_name, original_path_name):
    if io.verify_folder(os.path.dirname(result_path_name), create=True):
        affine = io.load_nifti(original_path_name)[1]
        io.save_nifti(img_reg, affine, result_path_name)


def apply_dfields(layout, ids, get_dfield_path_name, extra_info, img_save_type, skip_existing=False):
    """Warp the HUNT3 ``img_save_type`` image of every patient with ``get_dfield_path_name(id)``.

    Results are saved to ``layout.get_result_image_path_name(id, img_save_type, extra_info)``.
    """
    img_load_type, img_load_ext, binary = IMAGE_TYPES[img_save_type]
    for id in tqdm(ids):
        img_reg_path_name = layout.get_result_image_path_name(id, img_save_type, extra_info)
        if skip_existing and io.verify_file(img_reg_path_name):
            continue
        img_path_name = paths.get_patient_image_type_path_name(3, id, img_load_type, img_load_ext)
        img_reg = register_image(img_path_name, get_dfield_path_name(id), binary=binary)
        save_result(img_reg, img_reg_path_name, img_path_name)


def apply_all(layout, ids, get_dfield_path_name, extra_info, img_types=tuple(IMAGE_TYPES), skip_existing=False):
    for img_save_type in img_types:
        print(img_save_type)
        apply_dfields(layout, ids, get_dfield_path_name, extra_info, img_save_type, skip_existing=skip_existing)
