"""MEAN method: every validation patient gets the mean longitudinal dfield of the first ``nb_images`` reference patients."""
import numpy as np
from tqdm import tqdm

from brainpred import splits
from brainpred.methods.dfield import apply
from brainpred.methods.dfield.common import load_longitudinal_dfield
from brainpred.results import ResultsLayout
from brainpred.utils import io

LAYOUT = ResultsLayout("dfield_mean")
INAME_DFIELD = "MEAN_DFIELD_{0}.nii.gz"


def get_dfield_path_name(nb_images):
    # one dfield shared by every patient
    return f"{LAYOUT.get_images_path()}/dfields/{INAME_DFIELD.format(nb_images)}"


def compute_mean_dfield(nb_images, exclude_ids=None):
    patient_ids = splits.get_matching_ids()
    dfield_sum = None
    for i in tqdm(range(nb_images)):
        id = patient_ids[i]
        if exclude_ids is not None and id in exclude_ids:
            print(f"Excluding {id} from mean dfield computation")
            continue
        _dfield, affine = load_longitudinal_dfield(id)
        if dfield_sum is None:
            dfield_sum = _dfield
        else:
            dfield_sum += _dfield

    # NOTE: divides by nb_images even when some of them were excluded (original behaviour)
    dfield_sum /= nb_images
    return dfield_sum.astype(np.float32), affine


def create_dfields(nb_images_list=(100,)):
    exclude_ids = splits.get_excluded_patients() + splits.get_validation_ids()
    for nb_images in nb_images_list:
        dfield_mean, affine = compute_mean_dfield(nb_images, exclude_ids=exclude_ids)
        dfield_path_name = get_dfield_path_name(nb_images)
        io.create_folders(dfield_path_name.rsplit("/", 1)[0])
        io.save_nifti(dfield_mean, affine, dfield_path_name)


def apply_dfields(ids, nb_images_list=(100,), img_types=tuple(apply.IMAGE_TYPES), skip_existing=False):
    for nb_images in nb_images_list:
        print(f"-------- Registering images using MEAN DFIELD with {nb_images} images")
        apply.apply_all(LAYOUT, ids, lambda id: get_dfield_path_name(nb_images), (nb_images,), img_types, skip_existing)
