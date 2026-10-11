"""Rebuild 3D volumes from the externally produced 2D (axial PNG) diffusion-model predictions.

Port of ``approaches/diffusion_model/test_img_3D_reconstruction.ipynb``: each predicted slice
``<slices_dir>/<id>/<id>_1_T1_<layer>.png`` (8-bit, 160 x 224) is scaled to [0, 1] and pasted at
``[16:176, 3:227, layer]`` of a zero volume with the geometry of the patient's HUNT3 PREP_MNI
volume, for ``layer`` in ``[LAYER_START_2D, LAYER_END_2D)``.
"""
import os

import numpy as np
from PIL import Image

from brainpred import paths, splits
from brainpred.evaluation.statistics import get_available_ids
from brainpred.results import ResultsLayout
from brainpred.utils import io

METHOD_NAME = "diffusion_model"
EXTRA = "RES"
SLICE_NAME = "{0}_1_T1_{1}.png"


def get_results_layout():
    return ResultsLayout(METHOD_NAME)


def get_val_ids_2(postprocessed=True):
    """Patients with diffusion results (former ``pil.get_val_ids_2``: folders of results_postprocessed/)."""
    return get_available_ids(get_results_layout(), postprocessed)


def list_slice_ids(slices_dir):
    return sorted(e for e in os.listdir(slices_dir) if os.path.isdir(os.path.join(slices_dir, e)))


def create_3D_image(id, slices_dir):
    from brainpred.preprocessing import slices_2d

    img_hunt3, affine = io.load_nifti(slices_2d.get_registered_image_path_name(3, id))
    img_res = np.zeros_like(img_hunt3, dtype=np.float32)
    for layer in range(paths.LAYER_START_2D, paths.LAYER_END_2D):
        image = np.array(Image.open(f"{slices_dir}/{id}/{SLICE_NAME.format(id, layer)}"), dtype=np.float32) / 255
        img_res[16:176, 3:227, layer] = image
    return img_res, affine


def save_3D_images(slices_dir, ids=None):
    """Write ``<results_dir>/diffusion_model/results/<id>/<id>_0_T1_RES.nii.gz`` for every patient in ``slices_dir``."""
    for id in ids or list_slice_ids(slices_dir):
        img_res, affine = create_3D_image(id, slices_dir)
        img_path = get_results_layout().get_result_image_path_name(id, "T1", (EXTRA,))
        if io.verify_folder(os.path.dirname(img_path), create=True):
            io.save_nifti(img_res, affine, img_path)


def select_evaluation_ids(available_ids, n=83):
    """Patients evaluated in the thesis: all validation patients with results, completed up to ``n`` with the others."""
    val_ids = splits.get_validation_ids()
    in_val = sorted(set(available_ids) & set(val_ids))
    others = sorted(i for i in available_ids if i not in in_val)
    return sorted(in_val + others[:n - len(in_val)])
