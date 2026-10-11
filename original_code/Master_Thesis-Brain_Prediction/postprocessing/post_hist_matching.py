import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../'))
sys.path.append(base_path)

import numpy as np
import matplotlib.pyplot as plt

import utils.nifti_functions as nfc
import utils.functions as fc
from skimage.exposure import match_histograms

def save_histogram_match(path_name_ref, path_name_in, path_name_out, path_bmask_in=None, just_return=False):
    # Read the images
    img, aff = nfc.load_nifti(path_name_in)
    img_ref, _ = nfc.load_nifti(path_name_ref)

    if path_bmask_in is not None:
        bmask, aff = nfc.load_nifti(path_bmask_in)

        img[bmask == 0] = 0
        img[img < 0] = 0

    img = img.astype(np.float32) / np.max(img)


    # Match the histograms
    img = np.expand_dims(img, axis=-1)
    img_ref = np.expand_dims(img_ref, axis=-1)

    matched = match_histograms(img, img_ref, channel_axis=3)

    # fc.imgshow_3D("img", img, show=False)
    # fc.imgshow_3D("img_ref" , img_ref, show=False)
    # fc.imgshow_3D("matched" , matched, show=False)


    # plt.show()
    # Save the output
    if just_return:
        return matched[...,0]
    nfc.save_nifti(matched[...,0], aff, path_name_out)
