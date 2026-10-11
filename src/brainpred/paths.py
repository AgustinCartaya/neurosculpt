"""File naming conventions and path helpers for the longitudinal dataset.

Every image of patient ``id`` at visit ``hunt`` (3 or 4) lives in
``<data_dir>/HUNT<hunt>/<id>/`` and is named ``<id>_<hunt-3>_<TYPE>.<ext>``,
e.g. ``01234_0_T1_PREP.nii`` for the preprocessed HUNT3 scan.
See ``docs/data.md`` for the list of image types.
"""
from brainpred.config import ELASTIX_PARAMS_DIR, get_config
from brainpred.utils import io

# ---- Registration targets / libraries
REGISTRATION_TO_HUNT3 = "HUNT3"
REGISTRATION_TO_MNI = "MNI"
LIBRARY_ELASTIX = "ELASTIX"
LIBRARY_FREESURFER = "FREESURFER"

# ---- Elastix parameter files (configs/elastix/PAR_00<id>_<type>.txt)
REGISTRATION_PARAMETERS_FILES = "PAR_00{0}_{1}.txt"
PARAMETERS_FILE_RIGID = "RIG"
PARAMETERS_FILE_AFFINE = "AFF"
PARAMETERS_FILE_SIMILARITY = "SIM"
PARAMETERS_FILE_BSPLINE = "BSP"

# ---- Image names: {0}=patient id, {1}=hunt-3
FOLDER_HUNT = "HUNT{0}"
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
INAME_DFIELD_REG_IMAGE = "{0}_{1}_DFIELD_REGIMAGE_{2}_{3}.nii"
INAME_DFIELD = "{0}_{1}_DFIELD_{2}_{3}.nii"

# ---- Longitudinal (HUNT3 -> HUNT4) elastix registrations: {0}=id, {1}=type
INAME_LONGITUDINAL_REG = "{0}_0_{1}_0038.nii.gz"

# ---- 2D slices (StarGAN v2): {0}=id, {1}=hunt-3, {2}=slice index
LAYER_START_2D = 34
LAYER_END_2D = 157
INAME2D_T1 = "{0}_{1}_T1_{2}.png"
INAME2D_BMASK = "{0}_{1}_BMASK_{2}.png"
INAME2D_SEG_3 = "{0}_{1}_SEG_3_{2}.png"


# ---- Folders
def get_data_folder():
    return str(get_config().data_dir)


def get_hunt_folder(hunt):
    return f"{get_data_folder()}/{FOLDER_HUNT.format(hunt)}"


def get_patient_folder(hunt, id):
    return f"{get_hunt_folder(hunt)}/{id}"


def get_registration_param_folder():
    return str(ELASTIX_PARAMS_DIR)


def get_MNI_template_folder():
    return str(get_config().mni_template_dir)


def get_temp_folder():
    return str(get_config().temp_dir)


def get_freesurfer_license_folder():
    return str(get_config().freesurfer_license_dir)


def get_work_folder(*parts):
    """Folder for intermediate artefacts shared between methods, e.g. ``get_work_folder("similarity")``."""
    return "/".join([str(get_config().work_dir), *map(str, parts)])


def get_checkpoints_folder(*parts):
    return "/".join([str(get_config().checkpoints_dir), *map(str, parts)])


def get_figures_folder():
    return f"{get_config().results_dir}/figures"


# ---- Patient images
def get_patient_image_path_name(hunt, id, img_name):
    return f"{get_patient_folder(hunt, id)}/{img_name}"


def get_patient_image_type_path_name(hunt, id, img_type, extension="nii"):
    """Possible types: T1_BFC, T1_RAW, T1_MNI, T1_PREP, T1_SYNTHSR, SEG_36, SEG_96, SEG_3, BMASK_MNI, BMASK_NCRB_MNI."""
    return f"{get_patient_folder(hunt, id)}/{INAME.format(id, hunt - 3, img_type, extension)}"


def get_original_image_path_name(hunt, id):
    """Bias-field-corrected T1 if available, raw T1 otherwise."""
    path_bfc = get_patient_image_path_name(hunt, id, INAME_BFC_T1.format(id, hunt - 3))
    if io.verify_file(path_bfc):
        return path_bfc
    return get_patient_image_path_name(hunt, id, INAME_RAW_T1.format(id, hunt - 3))


def get_synthSR_image_path_name(hunt, id):
    return f"{get_patient_folder(hunt, id)}/{INAME_SYNTHSR_T1.format(id, hunt - 3)}"


def get_brain_mask_path_name(hunt, id, registration_to, no_cerebellum=False):
    if no_cerebellum:
        return f"{get_patient_folder(hunt, id)}/{INAME_BRAIN_MASK_NCRB.format(id, hunt - 3, registration_to)}"
    return f"{get_patient_folder(hunt, id)}/{INAME_BRAIN_MASK.format(id, hunt - 3, registration_to)}"


def get_preprocessed_image_path_name(hunt, id):
    return f"{get_patient_folder(hunt, id)}/{INAME_PREPROCESSED.format(id, hunt - 3)}"


def get_registered_to_template_image_path_name(hunt, id, registration_to):
    return f"{get_patient_folder(hunt, id)}/{INAME_REGISTERED_TEMPLATE.format(id, hunt - 3, registration_to)}"


def get_registered_template_dfield_path_name(hunt, id, registration_to):
    return f"{get_patient_folder(hunt, id)}/{DFIELD_NAME_REGISTERED_TEMPLATE.format(id, hunt - 3, registration_to)}"


def get_segmentation_path_name(hunt, id, nb_tissues=3, extension="nii.gz"):
    return f"{get_patient_folder(hunt, id)}/{INAME_SEGMENTATION.format(id, hunt - 3, nb_tissues, extension)}"


# ---- Longitudinal registrations (HUNT3 registered to HUNT4 with elastix, parameter file 0038)
def get_longitudinal_registration_folder():
    return get_work_folder("longitudinal_registrations")


def get_longitudinal_registration_path_name(id, type_img="T1"):
    """Possible types: T1, BMASK, DFIELD, SEG_96, SEG_36."""
    return f"{get_longitudinal_registration_folder()}/{INAME_LONGITUDINAL_REG.format(id, type_img)}"


# ---- 2D slices
def get_images_folder_2D(axial=False):
    return get_work_folder("slices_2D_axial" if axial else "slices_2D_sagittal")


def get_patient_folder_2D(hunt, id, axial=False):
    return f"{get_images_folder_2D(axial=axial)}/{FOLDER_HUNT.format(hunt)}/{id}"


def get_2D_image_path_name(hunt, id, layer, img_name, axial=False):
    return f"{get_patient_folder_2D(hunt, id, axial=axial)}/{img_name.format(id, hunt - 3, layer)}"


# ---- Templates and parameter files
def get_parameters_file_path_name(param_id, param_type):
    return f"{get_registration_param_folder()}/{REGISTRATION_PARAMETERS_FILES.format(param_id, param_type)}"


def get_T1_MNI_template_path_name(prep=False):
    if prep:
        return f"{get_MNI_template_folder()}/mni_icbm152_t1_tal_nlin_sym_09c_prep.nii"
    return f"{get_MNI_template_folder()}/mni_icbm152_t1_tal_nlin_sym_09c.nii"


def get_MNI_bmask_path_name():
    return f"{get_MNI_template_folder()}/mni_icbm152_t1_tal_nlin_sym_09c_mask.nii"


def get_genesis_weights_path_name():
    """Models Genesis pretrained weights, used by the perceptual loss of MPGAN."""
    return get_checkpoints_folder("genesis", "Genesis_Chest_CT.h5")


# ---- Loaders / savers
def load_image(hunt, id, img_name, transpose=False):
    return io.load_nifti(get_patient_image_path_name(hunt, id, img_name), transpose=transpose)


def load_original_image(hunt, id, transpose=False):
    return io.load_nifti(get_original_image_path_name(hunt, id), transpose=transpose)


def load_preprocessed_image(hunt, id, transpose=False):
    return io.load_nifti(get_preprocessed_image_path_name(hunt, id), transpose=transpose)


def save_image(img, affine, hunt, id, img_name, transpose=False):
    io.save_nifti(img, affine, get_patient_image_path_name(hunt, id, img_name), transpose=transpose)
