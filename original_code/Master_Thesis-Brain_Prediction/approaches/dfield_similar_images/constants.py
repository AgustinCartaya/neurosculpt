import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(base_path)

INAME_DFIELD = "{0}_IMGSIM_DFIELD_{1}_{2}.nii.gz"
INAME_RESULT_IMAGE = "{0}_0_{1}_{2}.nii.gz"

def get_images_path(absolute=True):
    if absolute:
        return f"{current_path}/images"
    return "images"

def get_dfield_path_name(id, sim_metric, nb_images, absolute=True):
    return f"{get_images_path(absolute)}/dfields/{id}/{INAME_DFIELD.format(id, sim_metric, nb_images)}"

def get_result_patient_folder(id, postprocessed=False, absolute=True):
    if postprocessed:
        return f"{get_images_path(absolute)}/results_postprocessed/{id}"
    return f"{get_images_path(absolute)}/results/{id}"

def get_result_image_path_name(id, type_img, extra_info, postprocessed=False, absolute=True):
    """possible type_img: T1, BMASK, DFIELD, SEG_96, SEG_36, SEG_3"""
    extra_info_str = "_".join([str(info) for info in extra_info])
    return f"{get_result_patient_folder(id, postprocessed, absolute)}/{INAME_RESULT_IMAGE.format(id, type_img, extra_info_str)}"

def get_statistics_path_name(statistics_name, postprocessed=False, absolute=True):
    if postprocessed:
        return f"{get_images_path(absolute)}/statistics_postprocessed/{statistics_name}.csv"
    return f"{get_images_path(absolute)}/statistics/{statistics_name}.csv"