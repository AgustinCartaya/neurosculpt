"""File-based postprocessing steps applied to predicted images (SynthSR, skull stripping, normalization, segmentation...)."""
import os
import shutil

import numpy as np

from brainpred.preprocessing import fastsurfer, normalization, skull_stripping, synthsr
from brainpred.preprocessing.external import run_command
from brainpred.preprocessing.cerebellum_mask import create_mask_no_cerebellum_from_seg_96
from brainpred.preprocessing.slices_2d import get_INV_registered_pfile_path_name
from brainpred.utils import io


def save_synthSR(path_name_in, path_name_out):
    return synthsr.run_synthsr(path_name_in, path_name_out)


def save_skull_stripping_freesurfer(path_name_in, mask_out, path_name_out=None):
    """Save the brain mask of ``path_name_in``; the skull-stripped image is kept only if ``path_name_out`` is given."""
    _path_name_out = path_name_out if path_name_out is not None else path_name_in + "rm.nii.gz"
    done = skull_stripping.run_synthstrip(path_name_in, mask_out, _path_name_out)
    if done and path_name_out is None:
        os.remove(_path_name_out)
    return done


def save_normalized_image(img_in_path_name, bmask_in_path_name, img_out_path_name):
    normalization.normalize_image_file(img_in_path_name, bmask_in_path_name, img_out_path_name)


def save_histogram_match(path_name_ref, path_name_in, path_name_out, path_bmask_in=None, just_return=False):
    """Match the histogram of ``path_name_in`` (optionally masked and scaled to [0, 1]) to ``path_name_ref``."""
    from skimage.exposure import match_histograms

    img, aff = io.load_nifti(path_name_in)
    img_ref, _ = io.load_nifti(path_name_ref)

    if path_bmask_in is not None:
        bmask, aff = io.load_nifti(path_bmask_in)
        img[bmask == 0] = 0
        img[img < 0] = 0

    img = img.astype(np.float32) / np.max(img)

    img = np.expand_dims(img, axis=-1)
    img_ref = np.expand_dims(img_ref, axis=-1)
    matched = match_histograms(img, img_ref, channel_axis=3)

    if just_return:
        return matched[..., 0]
    io.save_nifti(matched[..., 0], aff, path_name_out)


def segment_freesurfer(id, path_name_in, path_name_out, GPU=False):
    """FastSurfer 96-label segmentation of ``path_name_in``, saved as ``.mgz`` next to ``path_name_out``.

    Run :func:`transform_segmentation_to_patient_space` afterwards to obtain ``path_name_out`` (``.nii.gz``).
    """
    command = fastsurfer.get_fastsurfer_command(os.path.dirname(path_name_in), os.path.basename(path_name_in), id, gpu=GPU)
    done = run_command(command, f"Fastsurfer analysis done for {id}", encoding="utf-8")
    if done:
        if path_name_out.endswith("nii.gz"):
            path_name_out = path_name_out.replace("nii.gz", "mgz")
        shutil.copy(os.path.join(fastsurfer.get_fastsurfer_output_folder(id), "aparc.DKTatlas+aseg.deep.mgz"), path_name_out)
        fastsurfer.clean_temp_folder()
    return done


def transform_segmentation_to_patient_space(id, path_name, t1_img_path_name):
    """Convert ``path_name`` with ``.mgz`` extension to ``path_name`` (``.nii.gz``) in the space of ``t1_img_path_name``."""
    return fastsurfer.run_label2vol(path_name.replace("nii.gz", "mgz"), t1_img_path_name, path_name,
                                    f"Freesurfer transformation SEG done for {id}")


def create_mask_no_cerebellum(bmask_path_name, seg_96_path_name, output_path_name):
    create_mask_no_cerebellum_from_seg_96(bmask_path_name, seg_96_path_name, output_path_name)


def register_image_back(id, img_in_path_name, img_out_path_name):
    """Bring an image from the PREP_MNI space of the 2D methods back to the patient's preprocessed (MNI) space."""
    import itk

    param_obj = itk.ParameterObject.New()
    param_obj.AddParameterFile(get_INV_registered_pfile_path_name(3, id))

    img_moving = itk.imread(img_in_path_name, itk.F)
    itk.imwrite(itk.transformix_filter(img_moving, param_obj), img_out_path_name)
