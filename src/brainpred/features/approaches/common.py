"""Locate the postprocessed predictions of each method for the prediction feature tables."""
import os

from brainpred import splits
from brainpred.results import ResultsLayout
from brainpred.utils import io

# Configurations compared in the thesis: method (results folder name) -> extra_info of its result files.
FINAL_APPROACHES = {
    "dfield_similar_patches": [96, 10],
    "MGAN_paper_16": [46],
    "MPGAN_paper_16": [28],
    "MPGAN_paper_16_seg_loss": [52],
    "StarGAN_V2_sagital": ["BEST"],
    "diffusion_model": ["RES"],
}


def get_diffusion_model_ids():
    """The diffusion model was only evaluated on the patients with a postprocessed prediction."""
    results_folder = f"{ResultsLayout('diffusion_model').get_images_path()}/results_postprocessed"
    val_ids = [name for name in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, name))]
    val_ids.sort()
    return val_ids


def get_approach_ids(approach):
    if approach == "diffusion_model":
        return get_diffusion_model_ids()
    return splits.get_validation_ids()


def load_result_image(id, approach, type_img, extra_info):
    """Postprocessed prediction (``<TYPE>_POST``; ``<TYPE>_RAW`` for MPGAN_paper_16_seg_loss)."""
    img_sufix = "_RAW" if approach == "MPGAN_paper_16_seg_loss" else "_POST"
    path_name = ResultsLayout(approach).get_result_image_path_name(id, type_img + img_sufix, extra_info, postprocessed=True)
    return io.load_nifti(path_name)[0]
