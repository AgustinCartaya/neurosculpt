
import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(base_path)


import numpy as np
import project_images_loader as pil
from tqdm import tqdm

import itk

import constants as cts

def save_deformation_field_voxelmorph_MNI_space(id, pfile_number, trasnform=False):
    fixed_image = itk.imread(pil.get_registered_to_template_image_path_name(4, id, pil.REGISTRATION_TO_MNI), itk.F)
    moving_image = itk.imread(pil.get_registered_to_template_image_path_name(3, id, pil.REGISTRATION_TO_MNI), itk.F)

    # fixed_image = itk.imread(pil.get_preprocessed_image_path_name(4, id), itk.F)
    # moving_image = itk.imread(pil.get_preprocessed_image_path_name(3, id), itk.F)

    fixed_mask = itk.imread(pil.get_brain_mask_path_name(4, id, pil.REGISTRATION_TO_MNI), itk.UC)
    moving_mask = itk.imread(pil.get_brain_mask_path_name(3, id, pil.REGISTRATION_TO_MNI), itk.UC)

    param_obj_bspline = itk.ParameterObject.New()
    param_obj_bspline.AddParameterFile(cts.get_parameter_file_path_name(pfile_number, cts.PARAMETERS_FILE_BSPLINE))

    print("Registration started...")
    res_bspline, transform_parameters_bspline = itk.elastix_registration_method(fixed_image, moving_image,
                                                                            parameter_object=param_obj_bspline,
                                                                            moving_mask=moving_mask,
                                                                            fixed_mask=fixed_mask,
                                                                            log_to_console=False)
    
    deformation_field = itk.transformix_deformation_field(moving_image, transform_parameters_bspline)

    print("Saving")
    # path_name_img = f"{cts.images_folder}/{cts.INAME_REGISTERED.format(id, 0, pfile_number)}"
    # path_name_dfield = f"{cts.images_folder}/{cts.INAME_DFIELD.format(id, 0, pfile_number)}"
    itk.imwrite(res_bspline, cts.get_image_path_name(id, pfile_number, cts.INAME_REGISTERED), compression=True)
    itk.imwrite(deformation_field, cts.get_image_path_name(id, pfile_number, cts.INAME_DFIELD), compression=True)

    # saving param files
    # path_name_param_file = f"{cts.images_folder}/{cts.NAME_PARAMFILE.format(id, 0, pfile_number)}"
    transform_parameters_bspline.WriteParameterFile(transform_parameters_bspline, cts.get_trasformix_pfiles_path_name(id, pfile_number))

    if trasnform:
        # Register brain mask and segmentations
        moving_seg = itk.imread(pil.get_segmentation_path_name(3, id, simple=False), itk.US)
        moving_seg_simple = itk.imread(pil.get_segmentation_path_name(3, id, simple=True), itk.US)

        transform_parameters_bspline.SetParameter('FinalBSplineInterpolationOrder','0')
        mask_registered = itk.transformix_filter(moving_mask, transform_parameters_bspline)
        seg_registered = itk.transformix_filter(moving_seg, transform_parameters_bspline)
        seg_simple_registered = itk.transformix_filter(moving_seg_simple, transform_parameters_bspline)

        # path_mask_reg = f"{images_folder}/{MASK_REGISTERED.format(id, 0, pfile_number)}"
        # path_seg_reg = f"{images_folder}/{SEG_REGISTERED.format(id, 0, pfile_number)}"
        # path_seg_simple_reg = f"{images_folder}/{SEG_SIMPLE_REGISTERED.format(id, 0, pfile_number)}"

        itk.imwrite(mask_registered, cts.get_image_path_name(id, pfile_number, cts.INAME_MASK_REGISTERED), compression=True)
        itk.imwrite(seg_registered, cts.get_image_path_name(id, pfile_number, cts.INAME_SEG_REGISTERED), compression=True)
        itk.imwrite(seg_simple_registered, cts.get_image_path_name(id, pfile_number, cts.INAME_SEG_SIMPLE_REGISTERED), compression=True)



def transform_seg_and_mask(id, pfile_number):
    moving_mask = itk.imread(pil.get_brain_mask_path_name(3, id, pil.REGISTRATION_TO_MNI), itk.UC)
    moving_seg = itk.imread(pil.get_segmentation_path_name(3, id, simple=False), itk.US)
    moving_seg_simple = itk.imread(pil.get_segmentation_path_name(3, id, simple=True), itk.US)

    print("Aplying transformation...")

    param_obj_bspline = itk.ParameterObject.New()
    param_obj_bspline.AddParameterFile(cts.get_trasformix_pfiles_path_name(id, pfile_number))
    param_obj_bspline.SetParameter('FinalBSplineInterpolationOrder','0')

    # Register brain mask and segmentations
    mask_registered = itk.transformix_filter(moving_mask, param_obj_bspline)
    seg_registered = itk.transformix_filter(moving_seg, param_obj_bspline)
    seg_simple_registered = itk.transformix_filter(moving_seg_simple, param_obj_bspline)

    # path_mask_reg = f"{images_folder}/{MASK_REGISTERED.format(id, 0, pfile_number)}"
    # path_seg_reg = f"{images_folder}/{SEG_REGISTERED.format(id, 0, pfile_number)}"
    # path_seg_simple_reg = f"{images_folder}/{SEG_SIMPLE_REGISTERED.format(id, 0, pfile_number)}"

    itk.imwrite(mask_registered, cts.get_image_path_name(id, pfile_number, cts.INAME_MASK_REGISTERED), compression=True)
    itk.imwrite(seg_registered, cts.get_image_path_name(id, pfile_number, cts.INAME_SEG_REGISTERED), compression=True)
    itk.imwrite(seg_simple_registered, cts.get_image_path_name(id, pfile_number, cts.INAME_SEG_SIMPLE_REGISTERED), compression=True)



def start_registering():
    matching_ids = pil.get_matching_ids()
    progress_bar = tqdm(total=len(matching_ids), desc=f'Initializing...')

    for id in matching_ids:
        progress_bar.set_description(desc=f"Registering {id} HUNT{3} to HUNT{4}")
        progress_bar.update(1)

        # save_deformation_field_voxelmorph_MNI_space(id, pfile_number=cts.PFILE_0038, trasnform=False)
        transform_seg_and_mask(id, "0038")
        
start_registering()


