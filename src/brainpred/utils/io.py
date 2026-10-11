"""NIfTI I/O and filesystem helpers."""
import os

import nibabel as nib
import numpy as np
import SimpleITK as sitk


# ---- images
def load_nifti(path_name, transpose=False):
    """Return ``(data, (affine, header))``; pass the second element back to :func:`save_nifti`."""
    imag_nifti = nib.load(path_name)
    img_data = imag_nifti.get_fdata()
    if transpose:
        img_data = np.transpose(img_data, (1, 0, 2))
    return img_data, (imag_nifti.affine, imag_nifti.header)


def load_image_sitk(path_name):
    img_sitk = sitk.ReadImage(path_name)
    return sitk.GetArrayFromImage(img_sitk), img_sitk


def save_nifti(image_np, affine, img_path_name, transpose=False):
    if transpose:
        image_np = np.transpose(image_np, (1, 0, 2))
    img_nifti = nib.Nifti1Image(image_np, affine=affine[0], header=affine[1])
    nib.save(img_nifti, img_path_name)


# ---- filesystem
def create_folders(path):
    os.makedirs(path, exist_ok=True)


def verify_folder(path, create=False):
    """True if ``path`` is an existing folder; with ``create=True`` it is created first."""
    if create:
        create_folders(path)
    return os.path.isdir(path)


def verify_file(path_name):
    return os.path.isfile(path_name)
