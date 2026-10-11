"""SIM_IMG_REG method: like SIM_IMG, but each reference dfield is first registered (elastix) to the target patient's space."""
import os

import itk
import numpy as np

from brainpred import paths, splits
from brainpred.features.inter_patient import tissue_based as inter_patient_tissue_based
from brainpred.methods.dfield import apply
from brainpred.methods.dfield.common import compute_weighted_mean_dfield, get_excluded_reference_ids, load_longitudinal_dfield
from brainpred.results import ResultsLayout
from brainpred.utils import io

LAYOUT = ResultsLayout("dfield_similar_images_registered", "{0}_IMGSIM_DFIELD_{1}_{2}.nii.gz")
SIM_METRIC = "dice_mean"

# ---- elastix parameter files (configs/elastix/PAR_0038_<type>.txt)
PFILE_0038 = "38"
PARAMETERS_FILE_RIGID = "RIG"
PARAMETERS_FILE_AFFINE = "AFF"
PARAMETERS_FILE_BSPLINE = "BSP"
PARAMETERS_FILE_BSPLINE_FAST = "BSP_FAST"
REG_TYPES = (PARAMETERS_FILE_RIGID, PARAMETERS_FILE_AFFINE, PARAMETERS_FILE_BSPLINE, PARAMETERS_FILE_BSPLINE_FAST)

# ---- outputs of the target <- reference registrations: {0}=target id, {1}=reference id, {2}=reg type
PNAME_OBTAINED = "{0}_{1}_PAR_{2}.txt"
DFIELD_NAME_OBTAINED = "{0}_{1}_DFIELD_{2}.nii.gz"
INAME_OBTAINED = "{0}_{1}_{2}_{3}.nii.gz"


# ---- paths
def get_patient_similar_images_registration_folder(id):
    return f"{LAYOUT.get_images_path()}/similar_images_registrations/{id}"


def get_patient_obtained_parameter_file_path_name(id_target, id_data, registration_type):
    return f"{get_patient_similar_images_registration_folder(id_target)}/{PNAME_OBTAINED.format(id_target, id_data, registration_type)}"


def get_patient_registered_dfield_path_name(id_target, id_data, registration_type):
    return f"{get_patient_similar_images_registration_folder(id_target)}/{DFIELD_NAME_OBTAINED.format(id_target, id_data, registration_type)}"


def get_patient_registered_custom_path_name(id_target, id_data, imag_type, registration_type):
    """imag_type: T1, SEG_96"""
    return f"{get_patient_similar_images_registration_folder(id_target)}/{INAME_OBTAINED.format(id_target, id_data, imag_type, registration_type)}"


def get_registration_stages(reg_type):
    """Elastix stages run for ``reg_type``: rigid, then affine, then the B-spline variant."""
    stages = [PARAMETERS_FILE_RIGID]
    if reg_type in [PARAMETERS_FILE_AFFINE, PARAMETERS_FILE_BSPLINE, PARAMETERS_FILE_BSPLINE_FAST]:
        stages.append(PARAMETERS_FILE_AFFINE)
    if reg_type in [PARAMETERS_FILE_BSPLINE, PARAMETERS_FILE_BSPLINE_FAST]:
        stages.append(reg_type)
    return stages


def get_obtained_pfile_path_name_list(id_target, id_data, reg_type):
    return [get_patient_obtained_parameter_file_path_name(id_target, id_data, stage) for stage in get_registration_stages(reg_type)]


# ---- step 1: register each similar reference image to the target image
def save_registration_parameters_file(id_target, id_moving, obtained_pfile_path_name_list, reg_type):
    fixed_image = itk.imread(paths.get_registered_to_template_image_path_name(3, id_target, paths.REGISTRATION_TO_MNI), itk.F)
    moving_image = itk.imread(paths.get_registered_to_template_image_path_name(3, id_moving, paths.REGISTRATION_TO_MNI), itk.F)

    # NOTE: original behaviour kept: both masks belong to the target patient (HUNT4 as fixed, HUNT3 as moving mask)
    fixed_mask = itk.imread(paths.get_brain_mask_path_name(4, id_target, paths.REGISTRATION_TO_MNI), itk.UC)
    moving_mask = itk.imread(paths.get_brain_mask_path_name(3, id_target, paths.REGISTRATION_TO_MNI), itk.UC)

    param_obj = itk.ParameterObject.New()
    for stage in get_registration_stages(reg_type):
        param_obj.AddParameterFile(paths.get_parameters_file_path_name(PFILE_0038, stage))

    print("Registration started...")
    _, transform_parameters = itk.elastix_registration_method(fixed_image, moving_image,
                                                              parameter_object=param_obj,
                                                              moving_mask=moving_mask,
                                                              fixed_mask=fixed_mask,
                                                              log_to_console=False)
    transform_parameters.WriteParameterFiles(transform_parameters, obtained_pfile_path_name_list)


def load_reg_param_files(obtained_pfile_path_name_list):
    param_obj = itk.ParameterObject.New()
    for obtained_pfile_path_name in obtained_pfile_path_name_list:
        param_obj.AddParameterFile(obtained_pfile_path_name)
    return param_obj


def numpy2itk(np_array, itk_object):
    itk_image = itk.image_view_from_array(np_array)
    itk_image.CopyInformation(itk_object)
    return itk_image


# ---- step 2: move the reference's longitudinal dfield to the target space
def register_dfield(id_moving, reference_image, transform_parameters):
    """Transform each component of the reference's longitudinal dfield with ``transform_parameters``."""
    moving_dfield, dfield_aff = load_longitudinal_dfield(id_moving)
    moving_dfield = moving_dfield[:, :, :, 0, :]

    components = [itk.transformix_filter(numpy2itk(moving_dfield[:, :, :, axis], reference_image), transform_parameters)
                  for axis in range(3)]
    dfield_registered = np.stack([itk.array_from_image(c) for c in components], axis=3)
    dfield_registered = np.transpose(dfield_registered, (2, 1, 0, 3))
    dfield_registered = np.expand_dims(dfield_registered, axis=-2)
    return dfield_registered, dfield_aff


def transform_dfield_from_parameters_file(id_moving, obtained_pfile_path_name_list):
    param_obj = load_reg_param_files(obtained_pfile_path_name_list)
    moving_image_itk = itk.imread(paths.get_registered_to_template_image_path_name(3, id_moving, paths.REGISTRATION_TO_MNI), itk.F)
    return register_dfield(id_moving, moving_image_itk, param_obj)


# ---- optional step 3: move the reference image and segmentation (for visual checks)
def register_image_and_mask(id_moving, transform_parameters):
    img_path_name = paths.get_registered_to_template_image_path_name(3, id_moving, registration_to=paths.REGISTRATION_TO_MNI)
    seg_path_name = paths.get_segmentation_path_name(3, id_moving, nb_tissues=96)
    _, img_aff = io.load_nifti(img_path_name)
    _, seg_aff = io.load_nifti(seg_path_name)
    moving_img = itk.imread(img_path_name, itk.F)
    moving_seg = itk.imread(seg_path_name, itk.UC)

    img_reg = itk.transformix_filter(moving_img, transform_parameters)
    transform_parameters.SetParameter('FinalBSplineInterpolationOrder', '0')
    seg_reg = itk.transformix_filter(moving_seg, transform_parameters)

    img_reg = np.transpose(itk.array_from_image(img_reg), (2, 1, 0))
    seg_reg = np.transpose(itk.array_from_image(seg_reg), (2, 1, 0))
    return (img_reg, img_aff), (seg_reg, seg_aff)


def verify_parameters_files_path_name(obtained_pfile_path_name_list):
    return all(io.verify_file(p) for p in obtained_pfile_path_name_list)


def register_similar_images(ids, nb_images=5, reg_type=PARAMETERS_FILE_BSPLINE_FAST, sim_metric=SIM_METRIC, save_registered_images=False):
    """For each target in ``ids``: register its ``nb_images`` most similar references to it (step 1)
    and move their longitudinal dfields to the target space (step 2)."""
    sim_table_reader = inter_patient_tissue_based.FeaturesLoader(3, splits.get_matching_ids())
    exclude_ids = get_excluded_reference_ids(ids)
    for id in ids:
        if not io.verify_folder(get_patient_similar_images_registration_folder(id), create=True):
            continue
        print(f"-------- ID: {id}")
        print(f"---- Best {nb_images}: ")
        patient_sim_table = sim_table_reader.get_sorted_similarity_table(id, sim_metric, nb_images, exclude_ids=exclude_ids)

        for _, row in patient_sim_table.iterrows():
            id_reference = row['id']
            print(id_reference)
            obtained_pfile_path_name_list = get_obtained_pfile_path_name_list(id, id_reference, reg_type)
            if not verify_parameters_files_path_name(obtained_pfile_path_name_list):
                save_registration_parameters_file(id, id_reference, obtained_pfile_path_name_list, reg_type)

            reg_dfield_path_name = get_patient_registered_dfield_path_name(id, id_reference, reg_type)
            if not io.verify_file(reg_dfield_path_name):
                reg_dfield, dfield_aff = transform_dfield_from_parameters_file(id_reference, obtained_pfile_path_name_list)
                io.save_nifti(reg_dfield, dfield_aff, reg_dfield_path_name)

            if save_registered_images:
                params = load_reg_param_files(obtained_pfile_path_name_list)
                (img_reg, img_aff), (seg_reg, seg_aff) = register_image_and_mask(id_reference, params)
                io.save_nifti(img_reg, img_aff, get_patient_registered_custom_path_name(id, id_reference, "T1", reg_type))
                io.save_nifti(seg_reg, seg_aff, get_patient_registered_custom_path_name(id, id_reference, "SEG_96", reg_type))


# ---- target dfield
def create_dfields(ids, nb_images_list=(5,), reg_types=(PARAMETERS_FILE_BSPLINE_FAST,)):
    dice_mean_table_reader = inter_patient_tissue_based.FeaturesLoader(3, splits.get_matching_ids())
    exclude_ids = get_excluded_reference_ids(ids)

    for id in ids:
        print("id", id)
        for nb_images in nb_images_list:
            for reg_type in reg_types:
                dfield_path_name = LAYOUT.get_dfield_path_name(id, reg_type, nb_images)
                if not io.verify_file(dfield_path_name):
                    if io.verify_folder(os.path.dirname(dfield_path_name), create=True):
                        df_patient_sim_table = dice_mean_table_reader.get_sorted_similarity_table(id, SIM_METRIC, nb_images, exclude_ids=exclude_ids)
                        dfield_mean, affine = compute_weighted_mean_dfield(
                            df_patient_sim_table, lambda id_reference: io.load_nifti(get_patient_registered_dfield_path_name(id, id_reference, reg_type)))
                        io.save_nifti(dfield_mean, affine, dfield_path_name)


def apply_dfields(ids, nb_images_list=(5,), reg_types=(PARAMETERS_FILE_BSPLINE_FAST,), img_types=tuple(apply.IMAGE_TYPES), skip_existing=False):
    for nb_images in nb_images_list:
        for reg_type in reg_types:
            print(f"-------- Registering images using DFIELD {reg_type} with {nb_images} images")
            extra_info = (reg_type, nb_images)
            apply.apply_all(LAYOUT, ids, lambda id: LAYOUT.get_dfield_path_name(id, *extra_info), extra_info, img_types, skip_existing)
