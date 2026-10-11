import pathlib
import os

import utils.functions as fc
import utils.nifti_functions as nfc

import pandas as pd
import numpy as np


SERVER = False

# (not same patient) 01388: min: -1.0736, max: 4.6183, mean: 1.0000, std: 0.2147
# (same patient) 02408: min: -2.0260, max: 4.4923, mean: 1.0000, std: 0.1877
# (same patient) # 03139: min: -1.4722, max: 4.0456, mean: 1.0000, std: 0.0812
# (not same patient) 06611: min: -1.5924, max: 4.9465, mean: 1.0000, std: 0.1907
# (same patient) 02507: min: -0.7068, max: 3.6092, mean: 1.0000, std: 0.0665
EXCLUDED_PATIENTS = ['01388', '06611', '02408', '03139', '02507']


HUNTS = [3,4]
## CONSTANTS
## ---- Data folders
FOLDER_IMAGES = "images"
FOLDER_DATA_10 = "data_10"
FOLDER_DATA_703 = "data_703"
FOLDER_DATA = FOLDER_DATA_703

FOLDER_HUNT = "HUNT{0}"
FOLDER_PATIENT = "{0}"

## ---- Preprocessing
FOLDER_PREPROCESSING = "preprocessing"
FOLDER_REGISTRATION_PARAMETERS_ELASTIX = "registration_parameters_elastix"
FOLDER_MNI_TEMPLATE = "MNI_template"
FOLDER_FREESURFER_LICENSE = "freesurfer_license"

REGISTRATION_PARAMETERS_FILES = "PAR_00{0}_{1}.txt"
PARAMETERS_FILE_RIGID = "RIG"
PARAMETERS_FILE_AFFINE = "AFF"
PARAMETERS_FILE_SIMILARITY = "SIM"
PARAMETERS_FILE_BSPLINE = "BSP"

REGISTRATION_TO_HUNT3 = "HUNT3"
REGISTRATION_TO_MNI = "MNI"

LIBRARY_ELASTIX = "ELASTIX"
LIBRARY_FREESURFER = "FREESURFER"


## ---- Images names
INAME = "{0}_{1}_{2}.{3}"
INAME_BFC_T1 = "{0}_{1}_T1_BFC.nii"
INAME_RAW_T1 = "{0}_{1}_T1_RAW.nii"
INAME_SYNTHSR_T1 = "{0}_{1}_T1_SYNTHSR.nii"
INAME_BRAIN_MASK = "{0}_{1}_BMASK_{2}.nii"
INAME_BRAIN_MASK_NCRB = "{0}_{1}_BMASK_NCRB_{2}.nii.gz"

INAME_PREPROCESSED = "{0}_{1}_T1_PREP.nii"


INAME_SEGMENTATION = "{0}_{1}_SEG_{2}.{3}"

INAME_SEGMENTATION_3 = "{0}_{1}_SEG_3.{2}"
INAME_SEGMENTATION_96 = "{0}_{1}_SEG_96.{2}"
INAME_SEGMENTATION_36 = "{0}_{1}_SEG_36.{2}"
INAME_FASTSURFER_ORIG = "{0}_{1}_T1_FASTSURFER.{2}"

INAME_REGISTERED_TEMPLATE = "{0}_{1}_T1_{2}.nii"
DFIELD_NAME_REGISTERED_TEMPLATE = "{0}_{1}_DFIELD_{2}.lta"
# INAME_REGISTERED_BRAIN_MASK = "{0}_{1}_BMASK_REG_{2}_{3}_{4}.nii"

INAME_DFIELD_REG_IMAGE = "{0}_{1}_DFIELD_REGIMAGE_{2}_{3}.nii"
INAME_DFIELD = "{0}_{1}_DFIELD_{2}_{3}.nii"


## ---- 2D images
FOLDER_DATA_2D = "data_703_2D"
FOLDER_DATA_2D_AXIAL = "data_703_2D_axial"
LAYER_START_2D = 34
LAYER_END_2D = 157
INAME2D_T1 = "{0}_{1}_T1_{2}.png"
INAME2D_BMASK = "{0}_{1}_BMASK_{2}.png"
INAME2D_SEG_3 = "{0}_{1}_SEG_3_{2}.png"

def set_server_satet():
    global SERVER
    if os.path.exists("/cluster"):
        SERVER = True
    else:
        SERVER = False
    print(f"SERVER: {SERVER}")
set_server_satet()

## FUNCTIONS 
## ---- Paths
def get_base_folder(absolute=True):
    if SERVER:
        return "/cluster/projects/vc/data/mic/closed/MRI_HUNT/"
    if absolute:
        return str(pathlib.Path(__file__).parent.resolve()).replace(os.sep, "/") + "/"
    else:
        return ""


## -------- Folders
def get_hunt_folder(hunt, absolute=False):
    return f"{get_base_folder(absolute)}{FOLDER_IMAGES}/{FOLDER_DATA}/{FOLDER_HUNT.format(hunt)}"

def get_patient_folder(hunt, id, absolute=False):
    return f"{get_hunt_folder(hunt, absolute=absolute)}/{FOLDER_PATIENT.format(id)}"

def get_registration_param_folder(absolute=False):
    return f"{get_base_folder(absolute)}{FOLDER_PREPROCESSING}/{FOLDER_REGISTRATION_PARAMETERS_ELASTIX}"

def get_MNI_template_folder(absolute=False):
    return f"{get_base_folder(absolute)}{FOLDER_IMAGES}/{FOLDER_MNI_TEMPLATE}"

def get_data_reduced_folder(absolute=False):
    return f"{get_base_folder(absolute)}{FOLDER_IMAGES}/{FOLDER_DATA}"

def get_temp_folder(absolute=False):
    return f"{get_base_folder(absolute)}{FOLDER_IMAGES}/__temp"

def get_freesurfer_license_folder(absolute=False):
    return f"{get_base_folder(absolute)}{FOLDER_PREPROCESSING}/{FOLDER_FREESURFER_LICENSE}"

## -------- Images
## ------------ Preprocessing
def get_patient_image_path_name(hunt, id, img_name, absolute=False):
    return f"{get_patient_folder(hunt, id, absolute)}/{img_name}"

def get_patient_image_type_path_name(hunt, id, img_type, extension="nii", absolute=False):
    """current possible types: T1_BFC, T1_RAW, T1_MNI, T1_PREP, T1_SYNTHSR, SEG_36, SEG_96, SEG_3, BMASK_MNI, BMASK_NCRB_MNI"""
    return f"{get_patient_folder(hunt, id, absolute)}/{INAME.format(id, hunt-3, img_type, extension)}"

def get_original_image_path_name(hunt, id, absolute=False):
    path_bfc = get_patient_image_path_name(hunt, id, INAME_BFC_T1.format(id, hunt-3), absolute)
    if fc.verify_file(path_bfc):
        return path_bfc
    return get_patient_image_path_name(hunt, id, INAME_RAW_T1.format(id, hunt-3), absolute)

def get_synthSR_image_path_name(hunt, id, absolute=False):
    return f"{get_patient_folder(hunt, id, absolute)}/{INAME_SYNTHSR_T1.format(id, hunt-3)}"

def get_brain_mask_path_name(hunt, id, registration_to, no_cerebellum=False, absolute=False):
    if no_cerebellum:
        return f"{get_patient_folder(hunt, id, absolute)}/{INAME_BRAIN_MASK_NCRB.format(id, hunt-3, registration_to)}"
    return f"{get_patient_folder(hunt, id, absolute)}/{INAME_BRAIN_MASK.format(id, hunt-3, registration_to)}"

def get_preprocessed_image_path_name(hunt, id, absolute=False):
    return f"{get_patient_folder(hunt, id, absolute)}/{INAME_PREPROCESSED.format(id, hunt-3)}"

def get_registered_to_template_image_path_name(hunt, id, registration_to, absolute=False):
    return f"{get_patient_folder(hunt, id, absolute)}/{INAME_REGISTERED_TEMPLATE.format(id, hunt-3, registration_to)}"

def get_registered_template_dfield_path_name(hunt, id, registration_to, absolute=False):
    return f"{get_patient_folder(hunt, id, absolute)}/{DFIELD_NAME_REGISTERED_TEMPLATE.format(id, hunt-3, registration_to)}"

def get_segmentation_path_name(hunt, id, nb_tissues=3, extension="nii.gz", absolute=False):
    return f"{get_patient_folder(hunt, id, absolute)}/{INAME_SEGMENTATION.format(id, hunt-3, nb_tissues, extension)}"


# ---- 2D images
def get_images_folder_2D(axial=False, absolute=False):
    if axial:
        return f"{get_base_folder(absolute)}{FOLDER_IMAGES}/{FOLDER_DATA_2D_AXIAL}"
    return f"{get_base_folder(absolute)}{FOLDER_IMAGES}/{FOLDER_DATA_2D}"

def get_patient_folder_2D(hunt, id, axial=False, absolute=False):
    return f"{get_images_folder_2D(axial=axial, absolute=absolute)}/{FOLDER_HUNT.format(hunt)}/{FOLDER_PATIENT.format(id)}"

def get_2D_image_path_name(hunt, id, layer, img_name, axial=False, absolute=False):
    return f"{get_patient_folder_2D(hunt, id, axial=axial, absolute=absolute)}/{img_name.format(id, hunt-3, layer)}"

## ---- Registration
def get_parameters_file_path_name(param_id, param_type):
    return f"{get_registration_param_folder(param_type)}/{REGISTRATION_PARAMETERS_FILES.format(param_id, param_type)}"

def get_T1_MNI_template_path_name(prep=False, absolute=False):
    if prep:
        return f"{get_MNI_template_folder(absolute)}/mni_icbm152_t1_tal_nlin_sym_09c_prep.nii"
    return f"{get_MNI_template_folder(absolute)}/mni_icbm152_t1_tal_nlin_sym_09c.nii"

def get_MNI_bmask_path_name(absolute=False):
    return f"{get_MNI_template_folder(absolute)}/mni_icbm152_t1_tal_nlin_sym_09c_mask.nii"

## ---- Loaders
def load_image(hunt, id, img_name, transpose=False, absolute=False):
    return nfc.load_nifti(f"{get_patient_folder(hunt, id, absolute)}/{img_name}", transpose=transpose)

def load_original_image(hunt, id, transpose=False, absolute=False):
    return nfc.load_nifti(get_original_image_path_name(hunt, id, absolute), transpose=transpose)

def load_preprocessed_image(hunt, id, transpose=False, absolute=False):
    return nfc.load_nifti(get_preprocessed_image_path_name(hunt, id, absolute), transpose=transpose)



## --- savers
def save_image(img, affine, hunt, id, img_name, transpose=False, absolute=False):
    nfc.save_nifti(img, affine, f"{get_patient_folder(hunt, id, absolute)}/{img_name}", transpose=transpose)



## ---- test code paths
def get_genesis_weights_path_name():
    return f"{get_base_folder(True)}test_code/genesis_model/weights/Genesis_Chest_CT.h5"


def get_elastix_registered_image_path_name(id, type_img="T1", absolute=False):
    """possible types: T1, BMASK, DFIELD, SEG_96, SEG_36"""
    INAME_TEST_ELASTIX_REG = "{0}_0_{1}_0038.nii.gz"
    return f"{get_base_folder(absolute)}/test_code/dfield_approach_elastix/images/registrations/{INAME_TEST_ELASTIX_REG.format(id, type_img)}"


def get_matching_ids():
    def __get_patient_ids(hunt):
        # read folders
        hunt_folder = get_hunt_folder(hunt, absolute=True)
        possible_folders = os.listdir(hunt_folder)
        patient_folders = [e for e in possible_folders if os.path.isdir(os.path.join(hunt_folder, e))]
        return patient_folders

    patient_ids_HUNT3 = __get_patient_ids(3)
    patient_ids_HUNT4 = __get_patient_ids(4)
    matching_ids = list(set(patient_ids_HUNT3) & set(patient_ids_HUNT4))
    matching_ids.sort()
    return matching_ids



def get_val_ids_2():
    results_2D = f"{get_base_folder(absolute=True)}approaches/diffusion_model/images/results_postprocessed"
    val_ids_2 = [nombre for nombre in os.listdir(results_2D) if os.path.isdir(os.path.join(results_2D, nombre))]
    val_ids_2.sort()
    return val_ids_2



def get_save_image_path_name(img_name):
    # return f"C:/Users/Agustin/OneDrive - Université de Bourgogne/Master/S4/repport/final_repport/images/raw/{img_name}.png"
    return f"C:/Users/Agustin/Master/Master/S4/repport/MAIA_final_presentation/images/{img_name}.png"

def get_save_image_path_name_poster(img_name):
    # return f"C:/Users/Agustin/OneDrive - Université de Bourgogne/Master/S4/posters/MAIA/images/{img_name}.png"
    return f"C:/Users/Agustin/Master/Master/S4/repport/MAIA_final_presentation/images/{img_name}.png"
