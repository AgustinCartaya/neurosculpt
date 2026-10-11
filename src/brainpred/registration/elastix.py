"""Longitudinal HUNT3 -> HUNT4 B-spline registration with elastix (parameter file PAR_0038_BSP).

For each patient the MNI-registered HUNT3 image (moving) is registered to the
MNI-registered HUNT4 image (fixed), using both brain masks. Outputs, read by every
deformation-field method through ``paths.get_longitudinal_registration_path_name``::

    <work_dir>/longitudinal_registrations/{id}_0_T1_0038.nii.gz        registered HUNT3 image
    <work_dir>/longitudinal_registrations/{id}_0_DFIELD_0038.nii.gz    deformation field
    <work_dir>/longitudinal_registrations/{id}_0_PARAMFILE_0038.txt    transformix parameter file
    <work_dir>/longitudinal_registrations/{id}_0_{BMASK,SEG_96,SEG_36}_0038.nii.gz   transformed labels
"""
from brainpred import paths
from brainpred.utils import io

PFILE_ID = 38
NAME_PARAMFILE = "{0}_0_PARAMFILE_0038.txt"


def get_paramfile_path_name(id):
    return f"{paths.get_longitudinal_registration_folder()}/{NAME_PARAMFILE.format(id)}"


def _transform_seg_and_mask(id, param_obj):
    import itk

    moving_mask = itk.imread(paths.get_brain_mask_path_name(3, id, paths.REGISTRATION_TO_MNI), itk.UC)
    moving_seg_96 = itk.imread(paths.get_segmentation_path_name(3, id, 96), itk.US)
    moving_seg_36 = itk.imread(paths.get_segmentation_path_name(3, id, 36), itk.US)

    param_obj.SetParameter('FinalBSplineInterpolationOrder', '0')
    itk.imwrite(itk.transformix_filter(moving_mask, param_obj), paths.get_longitudinal_registration_path_name(id, "BMASK"), compression=True)
    itk.imwrite(itk.transformix_filter(moving_seg_96, param_obj), paths.get_longitudinal_registration_path_name(id, "SEG_96"), compression=True)
    itk.imwrite(itk.transformix_filter(moving_seg_36, param_obj), paths.get_longitudinal_registration_path_name(id, "SEG_36"), compression=True)


def register_longitudinal(id, transform=False):
    """Register HUNT3 to HUNT4 and save the registered image, deformation field and parameter file."""
    import itk

    io.verify_folder(paths.get_longitudinal_registration_folder(), create=True)

    fixed_image = itk.imread(paths.get_registered_to_template_image_path_name(4, id, paths.REGISTRATION_TO_MNI), itk.F)
    moving_image = itk.imread(paths.get_registered_to_template_image_path_name(3, id, paths.REGISTRATION_TO_MNI), itk.F)
    fixed_mask = itk.imread(paths.get_brain_mask_path_name(4, id, paths.REGISTRATION_TO_MNI), itk.UC)
    moving_mask = itk.imread(paths.get_brain_mask_path_name(3, id, paths.REGISTRATION_TO_MNI), itk.UC)

    param_obj_bspline = itk.ParameterObject.New()
    param_obj_bspline.AddParameterFile(paths.get_parameters_file_path_name(PFILE_ID, paths.PARAMETERS_FILE_BSPLINE))

    res_bspline, transform_parameters_bspline = itk.elastix_registration_method(fixed_image, moving_image,
                                                                                parameter_object=param_obj_bspline,
                                                                                moving_mask=moving_mask,
                                                                                fixed_mask=fixed_mask,
                                                                                log_to_console=False)
    deformation_field = itk.transformix_deformation_field(moving_image, transform_parameters_bspline)

    itk.imwrite(res_bspline, paths.get_longitudinal_registration_path_name(id, "T1"), compression=True)
    itk.imwrite(deformation_field, paths.get_longitudinal_registration_path_name(id, "DFIELD"), compression=True)
    transform_parameters_bspline.WriteParameterFile(transform_parameters_bspline, get_paramfile_path_name(id))

    if transform:
        _transform_seg_and_mask(id, transform_parameters_bspline)


def transform_seg_and_mask(id):
    """Apply a saved longitudinal transform to the HUNT3 brain mask and segmentations (nearest neighbour)."""
    import itk

    param_obj_bspline = itk.ParameterObject.New()
    param_obj_bspline.AddParameterFile(get_paramfile_path_name(id))
    _transform_seg_and_mask(id, param_obj_bspline)
