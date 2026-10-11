
import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../'))
sys.path.append(base_path)

import project_images_loader as pil
import itk
import numpy as np
import utils.nifti_functions as nfc
import utils.functions as fc

import subprocess



# # the registration is done to HUNT3 because for the test set i dont have imageges in the HUNT4 space 
# def save_registration_to_hunt3_elastix(id):
#     hunt = 4
#     fixed_image = itk.imread(pil.get_synthSR_image_path_name(3, id), itk.F)
#     moving_image = itk.imread(pil.get_synthSR_image_path_name(hunt, id), itk.F)

#     param_obj = itk.ParameterObject.New()
#     param_id = 26
#     param_obj.AddParameterFile(pil.get_parameters_file_path_name(param_id, pil.PARAMETERS_FILE_RIGID))
#     param_obj.AddParameterFile(pil.get_parameters_file_path_name(param_id, pil.PARAMETERS_FILE_SIMILARITY))


#     print("Registration started...")
#     # rigid registration
#     res_similarity, transform_parameters_rigid = itk.elastix_registration_method(fixed_image, moving_image,
#                                                                             parameter_object=param_obj,
#                                                                             log_to_console=False)
    
#     print("Saving")
#     itk.imwrite(res_similarity, pil.get_registered_image_path_name(hunt, id, pil.PARAMETERS_FILE_SIMILARITY, pil.REGISTRATION_TO_HUNT3, pil.LIBRARY_ELASTIX))


# def save_registration_to_MNI_elastix(hunt, id):
#     fixed_image = itk.imread(pil.get_T1_MNI_template_path_name(absolute=True), itk.F)
#     moving_image = itk.imread(pil.get_synthSR_image_path_name(hunt, id, absolute=True), itk.F)

#     param_obj = itk.ParameterObject.New()
#     param_id = 26
#     param_obj.AddParameterFile(pil.get_parameters_file_path_name(param_id, pil.PARAMETERS_FILE_RIGID))
#     param_obj.AddParameterFile(pil.get_parameters_file_path_name(param_id, pil.PARAMETERS_FILE_SIMILARITY))
#     # param_obj.AddParameterFile(pil.get_parameters_file_path_name(param_id, pil.PARAMETERS_FILE_AFFINE))


#     print("Registration started...")
#     # rigid registration
#     res_similarity, transform_parameters_rigid = itk.elastix_registration_method(fixed_image, moving_image,
#                                                                             parameter_object=param_obj,
#                                                                             # number_of_threads=4,
#                                                                             log_to_console=False)
    
    
#     # Register brain mask
#     transform_parameters_rigid.SetParameter('FinalBSplineInterpolationOrder','0')

#     print("Saving")
#     itk.imwrite(res_similarity, pil.get_registered_image_path_name(hunt, id, pil.PARAMETERS_FILE_SIMILARITY, pil.REGISTRATION_TO_MNI, pil.LIBRARY_ELASTIX))


def save_registration_to_hunt3_freesurfer(id):
    # mri_robust_register --mov 12082_1_T1_PREP.nii --dst 12082_0_T1_PREP.nii --lta dfield.lta --mapmov 12082_1_T1_REG_RIG_FREESURFER.nii --weights v1to2-weights.mgz --iscale --satit
    command = ["mri_robust_register", 
               "--mov", pil.get_synthSR_image_path_name(4, id, absolute=True), 
               "--dst", pil.get_synthSR_image_path_name(3, id, absolute=True), 
               "--mapmov", pil.get_registered_to_template_image_path_name(4, id, pil.REGISTRATION_TO_HUNT3, absolute=True), 
               "--iscale", 
               "--satit", 
            #    "--affine", 
               "--lta", pil.get_registered_template_dfield_path_name(4, id, pil.REGISTRATION_TO_MNI, absolute=True)]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        # Print the output
        # os.remove(pil.get_registered_template_dfield_path_name(4, id, pil.REGISTRATION_TO_MNI, absolute=True))
        print(f"Freesurfer registration done for HUNT{4} {id}")
        # print(result.stdout)
    else:
        # Print error message
        print("Error:", result.stderr)


def save_registration_to_MNI_freesurfer(hunt, id):
    # mri_robust_register --mov 12082_1_T1_PREP.nii --dst 12082_0_T1_PREP.nii --lta dfield.lta --mapmov 12082_1_T1_REG_RIG_FREESURFER.nii --weights v1to2-weights.mgz --iscale --satit
    if hunt == 3:
        path_name_dst = pil.get_T1_MNI_template_path_name(absolute=True)
    elif hunt == 4:
        path_name_dst = pil.get_registered_to_template_image_path_name(3, id, pil.REGISTRATION_TO_MNI, absolute=True)

    command = ["mri_robust_register", 
               "--mov", pil.get_synthSR_image_path_name(hunt, id, absolute=True), 
               "--dst", path_name_dst, 
               "--mapmov", pil.get_registered_to_template_image_path_name(hunt, id, pil.REGISTRATION_TO_MNI, absolute=True), 
               "--iscale", 
               "--satit", 
            #    "--affine", 
               "--lta", pil.get_registered_template_dfield_path_name(hunt, id, pil.REGISTRATION_TO_MNI, absolute=True)]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        # Print the output
        # os.remove(pil.get_registered_template_dfield_path_name(hunt, id, pil.REGISTRATION_TO_MNI, absolute=True))
        print(f"Freesurfer registration done for HUNT{hunt} {id}")
        # print(result.stdout)
    else:
        # Print error message
        print("Error:", result.stderr)
