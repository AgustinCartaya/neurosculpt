
import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../'))
sys.path.append(base_path)

import numpy as np

import project_images_loader as pil
import utils.nifti_functions as nfc
from skimage import exposure


def save_normalized_image(hunt, id):
    img_nsk, affine = pil.load_image(hunt, id, pil.INAME_REGISTERED_TEMPLATE.format(id, hunt-3, "MNI"))
    img_bmask, _ = pil.load_image(hunt, id, pil.INAME_BRAIN_MASK.format(id, hunt-3, "MNI"))

    # set background to 0
    img_nsk[img_bmask == 0] = 0
    img_nsk[img_nsk < 0] = 0

    # normalize image
    img_prep = img_nsk.astype(np.float32) / np.max(img_nsk)

    # save
    pil.save_image(img_prep, affine, hunt, id, pil.INAME_PREPROCESSED.format(id, hunt-3))





# def save_matched_histogram():
#     img_hunt4, affine = pil.load_image(4, 12082, "12082_1_T1_REG_SIM.nii")
#     img_hunt3, _ = pil.load_image(3, 12082, "12082_0_T1_PREP.nii")

#     img_hunt4_matched = exposure.match_histograms(img_hunt4, img_hunt3).astype(np.float32)

#     # save
#     pil.save_image(img_hunt4_matched, affine, 4, 12082, "12082_1_T1_HISTMATCH.nii")


# save_matched_histogram()
