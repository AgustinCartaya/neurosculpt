import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(base_path)


parameters_files_folder = f"{current_path}/parameters_files"
images_folder = f"{current_path}/images"


FOLDER_REGISTRATIONS = "registrations"
FOLDER_MEAN_DFIELDS = "mean_dfields"
FOLDER_SIMILAR_MEAN_DFIELDS = "similar_mean_dfields"
FOLDER_SIMILAR_REGISTERED_MEAN_DFIELDS = "similar_registered_mean_dfields"
FOLDER_PATCH_BASED_DFIELDS = "patches_based_dfield"
FOLDER_TISSUE_BASED_DFIELDS = "tissue_based_dfields"
# FOLDER_TEST = "test"

REGISTRATION_PARAMETERS_FILES = "PAR_{0}_{1}.txt"
PARAMETERS_FILE_BSPLINE = "BSP"
PARAMETERS_FILE_AFFINE= "AFF"
PARAMETERS_FILE_RIGID = "RIG"
PFILE_0038 = "0038"

INAME_REGISTERED = "{0}_{1}_T1_{2}.nii.gz"
INAME_DFIELD = "{0}_{1}_DFIELD_{2}.nii.gz"
NAME_PARAMFILE = "{0}_{1}_PARAMFILE_{2}.txt"

INAME_MASK_REGISTERED = "{0}_{1}_BMASK_{2}.nii.gz"
INAME_SEG_REGISTERED = "{0}_{1}_SEG_{2}_{3}.nii.gz"

INAME_DFIELD_MEAN = "MEAN_DFIELD_{0}_{1}.nii.gz"
INAME_SIMILAR_DFIELD_MEAN = "{0}_{1}_MEAN_DFIELD_{2}_{3}.nii.gz"
INAME_PATCH_BASED_DFIELD_MEAN = "{0}_{1}_{2}_PATCH_DFIELD_{3}_{4}.nii.gz"
INAME_TISSUE_BASED_DFIELD_MEAN = "{0}_T{1}_TISSUE_DFIELD_{2}_{3}.nii.gz"
INAME_SIMILAR_REGISTERED_DFIELD_MEAN = "{0}_{1}_{2}_DFIELD_{3}.nii.gz"



def get_parameter_file_path_name(pfile_number, registration_type=PARAMETERS_FILE_BSPLINE):
    return f"{parameters_files_folder}/{REGISTRATION_PARAMETERS_FILES.format(pfile_number, registration_type)}"

def get_image_path_name(id, pfile_number, iname, nb_tissues=None, subfolder=FOLDER_REGISTRATIONS):
    if nb_tissues is not None:
        return f"{images_folder}/{subfolder}/{iname.format(id, 0, nb_tissues, pfile_number)}"
    return f"{images_folder}/{subfolder}/{iname.format(id, 0, pfile_number)}"

def get_trasformix_pfiles_path_name(id, pfile_number, subfolder=FOLDER_REGISTRATIONS):
    return f"{images_folder}/{subfolder}/{NAME_PARAMFILE.format(id, 0, pfile_number)}"

def get_mean_dfield_path_name(pfile_number, nb_images, subfolder=FOLDER_MEAN_DFIELDS):
    return f"{images_folder}/{subfolder}/{INAME_DFIELD_MEAN.format(pfile_number, nb_images)}"

def get_similar_mean_dfield_path_name(id, sim_metric, pfile_number, nb_images, subfolder=FOLDER_SIMILAR_MEAN_DFIELDS):
    return f"{images_folder}/{subfolder}/{INAME_SIMILAR_DFIELD_MEAN.format(id, sim_metric, pfile_number, nb_images)}"

def get_similarity_csv_path_name(id):
    return f'{current_path}/approach_images_similar/similarities/{id}.csv'

def get_patch_based_dfield_path_name(id, sim_metric, patch_size, pfile_number, nb_images, subfolder=FOLDER_PATCH_BASED_DFIELDS):
    return f"{images_folder}/{subfolder}/{INAME_PATCH_BASED_DFIELD_MEAN.format(id, sim_metric, patch_size, pfile_number, nb_images)}"

def get_tissue_based_dfield_path_name(id, nb_tissues, pfile_number, nb_images, subfolder=FOLDER_TISSUE_BASED_DFIELDS):
    return f"{images_folder}/{subfolder}/{INAME_TISSUE_BASED_DFIELD_MEAN.format(id, nb_tissues, pfile_number, nb_images)}"

def get_similar_registered_mean_dfield_path_name(id, reg_type, sim_metric, nb_images, subfolder=FOLDER_SIMILAR_REGISTERED_MEAN_DFIELDS):
    return f"{images_folder}/{subfolder}/{INAME_SIMILAR_REGISTERED_DFIELD_MEAN.format(id, reg_type, sim_metric, nb_images)}"
