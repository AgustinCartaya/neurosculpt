import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(base_path)

INAME_RESULT_IMAGENC = "MNI_{0}.nii"
INAME_RESULT_IMAGE = "MNI_{0}.nii.gz"

def get_images_path(absolute=True):
    images_folder = "images"
    if absolute:
        return f"{current_path}/images"
    return images_folder

def get_result_image_path_name(type_img, absolute=True):
    if type_img == "T1":
        return f"{get_images_path(absolute)}/{INAME_RESULT_IMAGENC.format(type_img)}"
    return f"{get_images_path(absolute)}/{INAME_RESULT_IMAGE.format(type_img)}"
