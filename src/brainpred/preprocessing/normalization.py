"""Step 4: set the background to 0 and scale the MNI-registered image to [0, 1] (-> ``T1_PREP``)."""
import numpy as np

from brainpred import paths
from brainpred.utils import io


def normalize_with_brain_mask(img, bmask):
    """Zero everything outside ``bmask`` and the negative values, then divide by the maximum."""
    img[bmask == 0] = 0
    img[img < 0] = 0
    return img.astype(np.float32) / np.max(img)


def normalize_image_file(img_in_path_name, bmask_in_path_name, img_out_path_name):
    img, affine = io.load_nifti(img_in_path_name)
    bmask, _ = io.load_nifti(bmask_in_path_name)
    io.save_nifti(normalize_with_brain_mask(img, bmask), affine, img_out_path_name)


def save_normalized_image(hunt, id):
    normalize_image_file(paths.get_registered_to_template_image_path_name(hunt, id, paths.REGISTRATION_TO_MNI),
                         paths.get_brain_mask_path_name(hunt, id, paths.REGISTRATION_TO_MNI),
                         paths.get_preprocessed_image_path_name(hunt, id))
