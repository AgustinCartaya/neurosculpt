
import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../'))
sys.path.append(base_path)

import numpy as np
import utils.nifti_functions as nfc


def save_normalized_image(img_in_path_name, bmask_in_path_name, img_out_path_name):
    img_nsk, affine = nfc.load_nifti(img_in_path_name)
    img_bmask, _ = nfc.load_nifti(bmask_in_path_name)

    # set background to 0
    img_nsk[img_bmask == 0] = 0
    img_nsk[img_nsk < 0] = 0

    # normalize image
    img_prep = img_nsk.astype(np.float32) / np.max(img_nsk)

    # save
    nfc.save_nifti(img_prep, affine, img_out_path_name)





# def save_matched_histogram():
#     img_hunt4, affine = pil.load_image(4, 12082, "12082_1_T1_REG_SIM.nii")
#     img_hunt3, _ = pil.load_image(3, 12082, "12082_0_T1_PREP.nii")

#     img_hunt4_matched = exposure.match_histograms(img_hunt4, img_hunt3).astype(np.float32)

#     # save
#     pil.save_image(img_hunt4_matched, affine, 4, 12082, "12082_1_T1_HISTMATCH.nii")


# save_matched_histogram()
