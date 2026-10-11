import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../../'))
sys.path.append(base_path)

# Registration part
REGISTRATION_PARAMETERS_FILES = "PAR_{0}_{1}.txt"
PARAMETERS_FILE_BSPLINE_FAST = "BSP_FAST"
PARAMETERS_FILE_BSPLINE = "BSP"
PARAMETERS_FILE_AFFINE= "AFF"
PARAMETERS_FILE_RIGID = "RIG"
PFILE_0038 = "0038"

PNAME_OBTAINED = "{0}_{1}_PAR_{2}.txt"
DFIELD_NAME_OBTAINED = "{0}_{1}_DFIELD_{2}.nii.gz"
INAME_OBTAINED = "{0}_{1}_{2}_{3}.nii.gz"

def get_parameters_files_folder():
    return f"{current_path}/parameters_files"

def get_parameter_file_path_name(registration_type=PARAMETERS_FILE_BSPLINE, pfile_number=PFILE_0038):
    return f"{get_parameters_files_folder()}/{REGISTRATION_PARAMETERS_FILES.format(pfile_number, registration_type)}"

def get_pateint_similar_images_registration_folder_path(id):
    return f"{current_path}/images/{id}"

def get_patient_obtained_parameter_file_path_name(id_target, id_data, registration_type):
    return f"{get_pateint_similar_images_registration_folder_path(id_target)}/{PNAME_OBTAINED.format(id_target, id_data, registration_type)}"

def get_patient_registered_dfield_path_name(id_target, id_data, registration_type):
    return f"{get_pateint_similar_images_registration_folder_path(id_target)}/{DFIELD_NAME_OBTAINED.format(id_target, id_data, registration_type)}"

def get_patient_registered_custom_path_name(id_target, id_data, imag_type, registration_type):
    """imag_type: T1, REG_96"""
    return f"{get_pateint_similar_images_registration_folder_path(id_target)}/{INAME_OBTAINED.format(id_target, id_data, imag_type, registration_type)}"



def get_obtained_pfile_path_name_list(id_target, id_data, reg_type):
    obtained_pfile_path_name_list = []
    obtained_pfile_path_name_list.append(get_patient_obtained_parameter_file_path_name(id_target, id_data, PARAMETERS_FILE_RIGID))
    if reg_type in [PARAMETERS_FILE_AFFINE, PARAMETERS_FILE_BSPLINE, PARAMETERS_FILE_BSPLINE_FAST]:
        obtained_pfile_path_name_list.append(get_patient_obtained_parameter_file_path_name(id_target, id_data, PARAMETERS_FILE_AFFINE))
    if reg_type == PARAMETERS_FILE_BSPLINE:
        obtained_pfile_path_name_list.append(get_patient_obtained_parameter_file_path_name(id_target, id_data, PARAMETERS_FILE_BSPLINE))
    if reg_type == PARAMETERS_FILE_BSPLINE_FAST:
        obtained_pfile_path_name_list.append(get_patient_obtained_parameter_file_path_name(id_target, id_data, PARAMETERS_FILE_BSPLINE_FAST))
    return obtained_pfile_path_name_list


