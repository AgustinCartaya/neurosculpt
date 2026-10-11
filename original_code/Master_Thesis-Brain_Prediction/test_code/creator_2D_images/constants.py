
import sys
import os
import pathlib

current_path = os.path.dirname(__file__)
# base_path = os.path.abspath(os.path.join(current_path, '../../'))
# sys.path.append(base_path)

INAME = "{0}_{1}_T1_PREP_MNI.nii.gz"
INAME_PFILE = "{0}_{1}_PREP_MNI_DFIELD_AFF.txt"
INAME_INV_PFILE = "{0}_{1}_PREP_MNI_DFIELD_INV_AFF.txt"

INAME_BMASK = "{0}_{1}_BMASK_PREP_MNI.nii.gz"
INAME_SEG3 = "{0}_{1}_SEG_3_PREP_MNI.nii.gz"

def get_images_path(absolute=True):
    _base_path = str(pathlib.Path(__file__).parent.resolve()).replace(os.sep, "/") + "/"
    if absolute:
        return f"{_base_path}images_3D_preprocessed"
    return f"test_code/2D_images_creator/images_3D_preprocessed"

def get_patient_path(hunt, id, absolute=True):
    return f"{get_images_path(absolute)}/HUNT{hunt}/{id}"

def get_registered_image_path_name(hunt, id):
    return f"{get_patient_path(hunt, id)}/{INAME.format(id, hunt-3)}"

def get_registered_image_path_name2(hunt, id):
    return f"{get_patient_path(hunt, id)}/{INAME.format(id, hunt-3)}_test.nii"


def get_registered_pfile_path_name(hunt, id):
    return f"{get_patient_path(hunt, id)}/{INAME_PFILE.format(id, hunt-3)}"

def get_INV_registered_pfile_path_name(hunt, id):
    return f"{get_patient_path(hunt, id)}/{INAME_INV_PFILE.format(id, hunt-3)}"


def get_registered_bmask_path_name(hunt, id):
    return f"{get_patient_path(hunt, id)}/{INAME_BMASK.format(id, hunt-3)}"

def get_registered_seg3_path_name(hunt, id):
    return f"{get_patient_path(hunt, id)}/{INAME_SEG3.format(id, hunt-3)}"