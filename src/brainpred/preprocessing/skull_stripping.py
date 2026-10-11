"""Step 3: brain mask with FreeSurfer ``mri_synthstrip``."""
import os

from brainpred import paths
from brainpred.preprocessing.external import run_command


def run_synthstrip(path_name_in, mask_out, path_name_out):
    # mri_synthstrip -i <img> -o <stripped img> -m <mask>
    command = ["mri_synthstrip",
               "-i", path_name_in,
               "-o", path_name_out,
               "-m", mask_out]
    return run_command(command, f"Freesurfer skull stripping done for {path_name_in}")


def save_skull_stripping_freesurfer(hunt, id, registration_to):
    """Save the brain mask ``BMASK_<registration_to>``; the skull-stripped image itself is discarded."""
    if registration_to == paths.REGISTRATION_TO_HUNT3 and hunt == 3:
        path_name_in = paths.get_synthSR_image_path_name(hunt, id)
    else:
        path_name_in = paths.get_registered_to_template_image_path_name(hunt, id, registration_to)

    path_name_nsk = paths.get_patient_image_path_name(hunt, id, "nsk.nii")
    done = run_synthstrip(path_name_in, paths.get_brain_mask_path_name(hunt, id, registration_to), path_name_nsk)
    if done:
        os.remove(path_name_nsk)
    return done
