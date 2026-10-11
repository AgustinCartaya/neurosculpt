
import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(base_path)

import numpy as np

import project_images_loader as pil
import utils.nifti_functions as nfc
from skimage import exposure

import constants as cts


def save_normalized_image(hunt, id):
    img_nsk, affine = nfc.load_nifti(cts.get_registered_image_path_name(hunt, id))
    img_bmask, _ = nfc.load_nifti(cts.get_registered_bmask_path_name(hunt, id))

    # set background to 0
    img_nsk[img_bmask == 0] = 0
    img_nsk[img_nsk < 0] = 0

    # normalize image
    img_prep = img_nsk.astype(np.float32) / np.max(img_nsk)

    # save
    nfc.save_nifti(img_prep, affine, cts.get_registered_image_path_name(hunt, id))

