
import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(base_path)

import project_images_loader as pil
import itk
import numpy as np
import utils.nifti_functions as nfc
import utils.functions as fc
from tqdm import tqdm
import subprocess

import constants as cts

def save_registration_to_MNI_elastix(hunt, id, verify=True):
    if fc.verify_folder(cts.get_patient_path(hunt, id, absolute=False), create=True):
        if verify and not fc.verify_file(cts.get_registered_image_path_name(hunt, id)):

            param_id = 38

            if hunt == 3:
                path_name_fixed = pil.get_T1_MNI_template_path_name(prep=True, absolute=True)
            elif hunt == 4:
                path_name_fixed = cts.get_registered_image_path_name(3, id)

            fixed_image = itk.imread(path_name_fixed, itk.F)
            moving_image = itk.imread(pil.get_preprocessed_image_path_name(hunt, id, absolute=True), itk.F)

            param_obj = itk.ParameterObject.New()
            param_obj.AddParameterFile(pil.get_parameters_file_path_name(param_id, pil.PARAMETERS_FILE_AFFINE))

            # rigid registration
            res_similarity, transform_parameters = itk.elastix_registration_method(fixed_image, moving_image,
                                                                                    parameter_object=param_obj,
                                                                                    # number_of_threads=4,
                                                                                    log_to_console=False)
            
            
            # Register brain mask
            transform_parameters.WriteParameterFile(transform_parameters, cts.get_registered_pfile_path_name(hunt, id))
            
            itk.imwrite(res_similarity, cts.get_registered_image_path_name(hunt, id))


def save_transformation_mask_to_MNI(hunt, id, verify=True):
    param_obj = itk.ParameterObject.New()
    param_obj.AddParameterFile(cts.get_registered_pfile_path_name(hunt, id))

    param_obj.SetParameter('FinalBSplineInterpolationOrder','0')

    bmask_itk = itk.imread(pil.get_brain_mask_path_name(hunt, id, registration_to=pil.REGISTRATION_TO_MNI, absolute=True), itk.US)
    seg_itk = itk.imread(pil.get_segmentation_path_name(hunt, id, nb_tissues=3, absolute=True), itk.US)
    
    reg_bmask = itk.transformix_filter(bmask_itk, param_obj)
    reg_seg = itk.transformix_filter(seg_itk, param_obj)

    itk.imwrite(reg_bmask, cts.get_registered_bmask_path_name(hunt, id))
    itk.imwrite(reg_seg, cts.get_registered_seg3_path_name(hunt, id))

