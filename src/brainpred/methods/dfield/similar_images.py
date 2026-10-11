"""SIM_IMG method: weighted mean of the longitudinal dfields of the most similar reference patients (whole-image similarity)."""
import os

from brainpred import splits
from brainpred.features.inter_patient import tissue_based as inter_patient_tissue_based
from brainpred.features.inter_patient import whole_brain_based as inter_patient_whole_brain_based
from brainpred.methods.dfield import apply
from brainpred.methods.dfield.common import compute_weighted_mean_dfield, get_excluded_reference_ids, load_longitudinal_dfield
from brainpred.results import ResultsLayout
from brainpred.utils import io

LAYOUT = ResultsLayout("dfield_similar_images", "{0}_IMGSIM_DFIELD_{1}_{2}.nii.gz")
SIM_METRICS = ('ssim', 'dice_mean')


def compute_similar_images_mean_dfield(sim_metric, nb_images, df_metrics):
    df_metrics = df_metrics.sort_values(sim_metric, ascending=False)
    df_best_metrics = df_metrics.head(nb_images).copy()
    df_best_metrics['weight'] = df_best_metrics[sim_metric] / df_best_metrics[sim_metric].sum()
    return compute_weighted_mean_dfield(df_best_metrics, load_longitudinal_dfield)


def get_similarity_loader(sim_metric, matching_ids):
    """``ssim``: whole-brain HUNT3 similarity; ``dice_mean``: mean Dice of the 3-tissue segmentations."""
    if sim_metric == 'ssim':
        return inter_patient_whole_brain_based.FeaturesLoader(matching_ids)
    return inter_patient_tissue_based.FeaturesLoader(3, matching_ids)


def create_dfields(ids, nb_images_list=(100,), sim_metrics=('dice_mean',)):
    matching_ids = splits.get_matching_ids()
    loaders = {sim_metric: get_similarity_loader(sim_metric, matching_ids) for sim_metric in sim_metrics}
    exclude_ids = get_excluded_reference_ids(ids)

    for id in ids:
        print("id", id)
        for nb_images in nb_images_list:
            for sim_metric in sim_metrics:
                dfield_path_name = LAYOUT.get_dfield_path_name(id, sim_metric.upper(), nb_images)
                if not io.verify_file(dfield_path_name):
                    if io.verify_folder(os.path.dirname(dfield_path_name), create=True):
                        df_patient_sim_table = loaders[sim_metric].get_patient_similarity_table(id, exclude_ids=exclude_ids)
                        dfield_mean, affine = compute_similar_images_mean_dfield(sim_metric, nb_images, df_patient_sim_table)
                        io.save_nifti(dfield_mean, affine, dfield_path_name)


def apply_dfields(ids, nb_images_list=(100,), sim_metrics=('dice_mean',), img_types=tuple(apply.IMAGE_TYPES), skip_existing=True):
    for nb_images in nb_images_list:
        for sim_metric in sim_metrics:
            print(f"-------- Registering images using DFIELD {sim_metric} with {nb_images} images")
            extra_info = (sim_metric.upper(), nb_images)
            apply.apply_all(LAYOUT, ids, lambda id: LAYOUT.get_dfield_path_name(id, *extra_info), extra_info, img_types, skip_existing)
