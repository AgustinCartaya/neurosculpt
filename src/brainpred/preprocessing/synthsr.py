"""Step 1: SynthSR (FreeSurfer ``mri_synthsr``) on the original T1 scans."""
from brainpred import paths
from brainpred.preprocessing.external import run_command
from brainpred.utils import io


def run_synthsr(path_name_in, path_name_out):
    command = ["mri_synthsr",
               "--i", path_name_in,
               "--o", path_name_out,
               "--threads", "1"]
    return run_command(command, f"Freesurfer synthSR done for \n {path_name_in}")


def save_synthSR(hunt, id, verify=False):
    if verify and io.verify_file(paths.get_synthSR_image_path_name(hunt, id)):
        print(f"SynthSR already done for HUNT{hunt} {id}")
        return True
    return run_synthsr(paths.get_original_image_path_name(hunt, id), paths.get_synthSR_image_path_name(hunt, id))
