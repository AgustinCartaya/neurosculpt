import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(base_path)

INAME_DFIELD = "MEAN_DFIELD_{0}.nii.gz"
INAME_RESULT_IMAGE = "{0}_0_{1}_{2}.nii.gz"

def get_images_path(absolute=True):
    if absolute:
        return f"{current_path}/images"
    return "images"

def get_dfield_path_name(nb_images, absolute=True):
    return f"{get_images_path(absolute)}/dfields/{INAME_DFIELD.format(nb_images)}"

def get_result_image_path_name(id, type_img, extra_info, absolute=True):
    """possible type_img: T1, BMASK, DFIELD, SEG_96, SEG_36, SEG_3"""
    extra_info_str = "_".join([str(info) for info in extra_info])
    return f"{get_images_path(absolute)}/results/{id}/{INAME_RESULT_IMAGE.format(id, type_img, extra_info_str)}"

def get_statistics_path_name(statistics_name, absolute=True):
    return f"{get_images_path(absolute)}/statistics/{statistics_name}.csv"