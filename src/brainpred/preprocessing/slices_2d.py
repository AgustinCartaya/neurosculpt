"""2D slices for StarGAN v2: affine "PREP_MNI" registration of the preprocessed images and PNG slice export.

Pipeline (per patient, HUNT3 before HUNT4):

1. ``register``: affine elastix registration (PAR_0038_AFF) of ``T1_PREP`` to the preprocessed
   MNI template (HUNT3) or to the registered HUNT3 image (HUNT4) -> ``T1_PREP_MNI`` + parameter file;
2. ``transform``: apply that transform to ``BMASK_MNI`` and ``SEG_3`` (nearest neighbour);
3. ``normalize``: zero background, scale to [0, 1] (in place);
4. ``inverse``: inverse transform parameter file (PREP_MNI -> patient space), used to bring
   StarGAN / diffusion predictions back (see ``postprocessing.steps.register_image_back``);
5. ``slices``: export slices ``LAYER_START_2D..LAYER_END_2D-1`` of T1, BMASK and SEG_3 as PNG to
   ``paths.get_2D_image_path_name(hunt, id, layer, paths.INAME2D_*, axial=...)``.

3D outputs live in ``<work_dir>/prep_mni_3D/HUNT<h>/<id>/`` (former ``test_code/creator_2D_images/images_3D_preprocessed``).
The functions below keep the names of the original ``test_code.creator_2D_images.constants`` module.
"""
import numpy as np
from tqdm import tqdm

from brainpred import paths
from brainpred.preprocessing.normalization import normalize_with_brain_mask
from brainpred.utils import io

# ---- 3D PREP_MNI images: {0}=id, {1}=hunt-3
INAME = "{0}_{1}_T1_PREP_MNI.nii.gz"
INAME_PFILE = "{0}_{1}_PREP_MNI_DFIELD_AFF.txt"
INAME_INV_PFILE = "{0}_{1}_PREP_MNI_DFIELD_INV_AFF.txt"
INAME_BMASK = "{0}_{1}_BMASK_PREP_MNI.nii.gz"
INAME_SEG3 = "{0}_{1}_SEG_3_PREP_MNI.nii.gz"

ELASTIX_PARAM_ID = 38


def get_images_path():
    return paths.get_work_folder("prep_mni_3D")


def get_patient_path(hunt, id):
    return f"{get_images_path()}/HUNT{hunt}/{id}"


def get_registered_image_path_name(hunt, id):
    return f"{get_patient_path(hunt, id)}/{INAME.format(id, hunt - 3)}"


def get_registered_pfile_path_name(hunt, id):
    return f"{get_patient_path(hunt, id)}/{INAME_PFILE.format(id, hunt - 3)}"


def get_INV_registered_pfile_path_name(hunt, id):
    return f"{get_patient_path(hunt, id)}/{INAME_INV_PFILE.format(id, hunt - 3)}"


def get_registered_bmask_path_name(hunt, id):
    return f"{get_patient_path(hunt, id)}/{INAME_BMASK.format(id, hunt - 3)}"


def get_registered_seg3_path_name(hunt, id):
    return f"{get_patient_path(hunt, id)}/{INAME_SEG3.format(id, hunt - 3)}"


def _get_fixed_image_path_name(hunt, id):
    if hunt == 3:
        return paths.get_T1_MNI_template_path_name(prep=True)
    elif hunt == 4:
        return get_registered_image_path_name(3, id)
    raise ValueError(f"Unknown hunt {hunt}")


# ---- Step 1: affine registration to the preprocessed MNI template
def save_registration_to_MNI_elastix(hunt, id, verify=True):
    import itk

    io.verify_folder(get_patient_path(hunt, id), create=True)
    if verify and io.verify_file(get_registered_image_path_name(hunt, id)):
        return

    fixed_image = itk.imread(_get_fixed_image_path_name(hunt, id), itk.F)
    moving_image = itk.imread(paths.get_preprocessed_image_path_name(hunt, id), itk.F)

    param_obj = itk.ParameterObject.New()
    param_obj.AddParameterFile(paths.get_parameters_file_path_name(ELASTIX_PARAM_ID, paths.PARAMETERS_FILE_AFFINE))

    res_affine, transform_parameters = itk.elastix_registration_method(fixed_image, moving_image,
                                                                       parameter_object=param_obj,
                                                                       log_to_console=False)

    transform_parameters.WriteParameterFile(transform_parameters, get_registered_pfile_path_name(hunt, id))
    itk.imwrite(res_affine, get_registered_image_path_name(hunt, id))


# ---- Step 2: transform brain mask and 3-tissue segmentation
def save_transformation_mask_to_MNI(hunt, id):
    import itk

    param_obj = itk.ParameterObject.New()
    param_obj.AddParameterFile(get_registered_pfile_path_name(hunt, id))
    param_obj.SetParameter('FinalBSplineInterpolationOrder', '0')

    bmask_itk = itk.imread(paths.get_brain_mask_path_name(hunt, id, registration_to=paths.REGISTRATION_TO_MNI), itk.US)
    seg_itk = itk.imread(paths.get_segmentation_path_name(hunt, id, nb_tissues=3), itk.US)

    reg_bmask = itk.transformix_filter(bmask_itk, param_obj)
    reg_seg = itk.transformix_filter(seg_itk, param_obj)

    itk.imwrite(reg_bmask, get_registered_bmask_path_name(hunt, id))
    itk.imwrite(reg_seg, get_registered_seg3_path_name(hunt, id))


# ---- Step 3: normalization (overwrites the registered image)
def save_normalized_image(hunt, id):
    img, affine = io.load_nifti(get_registered_image_path_name(hunt, id))
    bmask, _ = io.load_nifti(get_registered_bmask_path_name(hunt, id))
    io.save_nifti(normalize_with_brain_mask(img, bmask), affine, get_registered_image_path_name(hunt, id))


# ---- Step 4: inverse transform parameter file
def find_inverse_registration_param_files(hunt, id):
    """Elastix inverse: register the fixed image to itself starting from the forward transform."""
    import itk

    fixed_image = itk.imread(_get_fixed_image_path_name(hunt, id), itk.F)

    param_obj = itk.ParameterObject.New()
    param_obj.AddParameterFile(paths.get_parameters_file_path_name(ELASTIX_PARAM_ID, paths.PARAMETERS_FILE_AFFINE))

    _, transform_parameters = itk.elastix_registration_method(fixed_image, fixed_image,
                                                              parameter_object=param_obj,
                                                              initial_transform_parameter_file_name=get_registered_pfile_path_name(hunt, id),
                                                              log_to_console=True)
    transform_parameters.SetParameter('InitialTransformParameterFileName', 'NoInitialTransform')
    transform_parameters.WriteParameterFile(transform_parameters, get_INV_registered_pfile_path_name(hunt, id))


# ---- Step 5: PNG slices
def get_2D_slides_from_3D_volume(img_3D, start_slide, end_slide, axis=0):
    img_2D_list = []
    for slide in range(start_slide, end_slide):
        if axis == 0:
            img_2D = img_3D[slide:slide + 1]
        elif axis == 1:
            img_2D = img_3D[:, slide:slide + 1]
        elif axis == 2:
            img_2D = img_3D[:, :, slide:slide + 1]
        img_2D_list.append(np.squeeze(img_2D))
    return img_2D_list


def create_and_save_2D_images(img_3D, img_2d_path_name, scale_intensity=True, axis=0):
    """Save slices as 8-bit PNG; ``img_2d_path_name`` is a format string receiving the slice index."""
    from PIL import Image

    img_2D_list = get_2D_slides_from_3D_volume(img_3D, paths.LAYER_START_2D, paths.LAYER_END_2D, axis=axis)
    for i, img_2D in enumerate(img_2D_list):
        current_layer = paths.LAYER_START_2D + i
        img_to_save = img_2D
        if scale_intensity:
            img_to_save *= 255.0
        Image.fromarray(img_to_save.astype(dtype=np.uint8)).save(img_2d_path_name.format(current_layer))


def create_and_save_2D_images_patient(hunt, id, axis=2):
    """axis=2 writes to the axial 2D folder, any other axis to the sagittal one."""
    axial = axis == 2
    img_3D = io.load_nifti(get_registered_image_path_name(hunt, id))[0]
    bmask_3D = io.load_nifti(get_registered_bmask_path_name(hunt, id))[0]
    seg_3D = io.load_nifti(get_registered_seg3_path_name(hunt, id))[0]

    if io.verify_folder(paths.get_patient_folder_2D(hunt, id, axial=axial), create=True):
        create_and_save_2D_images(img_3D, paths.get_2D_image_path_name(hunt, id, "{0}", paths.INAME2D_T1, axial=axial), axis=axis)
        create_and_save_2D_images(bmask_3D, paths.get_2D_image_path_name(hunt, id, "{0}", paths.INAME2D_BMASK, axial=axial), axis=axis)
        create_and_save_2D_images(seg_3D, paths.get_2D_image_path_name(hunt, id, "{0}", paths.INAME2D_SEG_3, axial=axial), scale_intensity=False, axis=axis)


# ---- Analysis used to choose LAYER_START_2D / LAYER_END_2D
def load_mask_prep_mni(id, hunt):
    img, _ = io.load_nifti(get_registered_image_path_name(hunt, id))
    return np.where(img > 0.01, 1, 0)


def find_start_end_layer(img, min_tissue=0.1, axis=0):
    """First and last slice (along axis 0) whose fraction of brain voxels is >= ``min_tissue``."""
    start = 0
    end = img.shape[axis]
    max_area = np.prod(np.delete(img.shape, axis))

    for i in range(img.shape[axis]):
        if np.sum(img[i]) / max_area >= min_tissue:
            start = i
            break

    for i in range(img.shape[axis] - 1, 0, -1):
        if np.sum(img[i]) / max_area >= min_tissue:
            end = i
            break
    return start, end


def create_start_end_list(matching_ids, min_tissue=0.15, axis=0):
    """Per-scan first/last brain slice. With PREP_MNI images, min_tissue=0.15 and axis=0 the median was
    [34, 158] (min [32, 157], max [35, 160]), from which LAYER_START_2D=34 and LAYER_END_2D=157 were chosen."""
    layer_start_end = np.zeros((len(matching_ids) * 2, 2))
    for i, id in enumerate(tqdm(matching_ids)):
        for j, hunt in enumerate([3, 4]):
            mask = load_mask_prep_mni(id, hunt)
            layer_start_end[i * 2 + j] = find_start_end_layer(mask, min_tissue=min_tissue, axis=axis)
    return layer_start_end
