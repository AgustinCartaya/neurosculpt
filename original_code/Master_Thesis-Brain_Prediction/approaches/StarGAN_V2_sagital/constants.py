import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(base_path)

INAME_DFIELD = "{0}_TISSUESIM_DFIELD_{1}_{2}.nii.gz"
INAME_RESULT_IMAGE = "{0}_0_{1}_{2}.nii.gz"

def get_images_path(prep_mni=True, absolute=True):
    images_folder = "images_NO_PREP_MNI" if not prep_mni else "images"
    if absolute:
        return f"{current_path}/{images_folder}"
    return images_folder

def get_result_patient_folder(id, postprocessed=False, prep_mni=True, absolute=True):
    if postprocessed:
        return f"{get_images_path(prep_mni, absolute)}/results_postprocessed/{id}"
    return f"{get_images_path(prep_mni, absolute)}/results/{id}"


def get_result_image_path_name(id, type_img, extra_info, postprocessed=False, prep_mni=True, absolute=True):
    """possible type_img: T1, BMASK, DFIELD, SEG_96, SEG_36, SEG_3"""
    extra_info_str = "_".join([str(info) for info in extra_info])
    return f"{get_result_patient_folder(id, postprocessed, prep_mni, absolute)}/{INAME_RESULT_IMAGE.format(id, type_img, extra_info_str)}"


def get_statistics_path_name(statistics_name, postprocessed=False, prep_mni=True, absolute=True):
    if postprocessed:
        return f"{get_images_path(prep_mni, absolute)}/statistics_postprocessed/{statistics_name}.csv"
    return f"{get_images_path(prep_mni, absolute)}/statistics/{statistics_name}.csv"