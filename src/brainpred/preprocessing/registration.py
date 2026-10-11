"""Step 2: rigid registration of the SynthSR images to MNI space with FreeSurfer ``mri_robust_register``.

HUNT3 is registered to the MNI template and HUNT4 is registered to the already
registered HUNT3 image, so for a given patient HUNT3 must be processed first.
"""
from brainpred import paths
from brainpred.preprocessing.external import run_command


def run_robust_register(path_name_mov, path_name_dst, path_name_mapmov, path_name_lta):
    # mri_robust_register --mov <mov> --dst <dst> --mapmov <out> --iscale --satit --lta <lta>
    command = ["mri_robust_register",
               "--mov", path_name_mov,
               "--dst", path_name_dst,
               "--mapmov", path_name_mapmov,
               "--iscale",
               "--satit",
               "--lta", path_name_lta]
    return run_command(command, f"Freesurfer registration done for {path_name_mov}")


def save_registration_to_MNI_freesurfer(hunt, id):
    if hunt == 3:
        path_name_dst = paths.get_T1_MNI_template_path_name()
    elif hunt == 4:
        path_name_dst = paths.get_registered_to_template_image_path_name(3, id, paths.REGISTRATION_TO_MNI)
    else:
        raise ValueError(f"Unknown hunt {hunt}")

    return run_robust_register(paths.get_synthSR_image_path_name(hunt, id),
                               path_name_dst,
                               paths.get_registered_to_template_image_path_name(hunt, id, paths.REGISTRATION_TO_MNI),
                               paths.get_registered_template_dfield_path_name(hunt, id, paths.REGISTRATION_TO_MNI))


def save_registration_to_hunt3_freesurfer(id):
    """Register HUNT4 to HUNT3 in patient space (not part of the final pipeline)."""
    return run_robust_register(paths.get_synthSR_image_path_name(4, id),
                               paths.get_synthSR_image_path_name(3, id),
                               paths.get_registered_to_template_image_path_name(4, id, paths.REGISTRATION_TO_HUNT3),
                               paths.get_registered_template_dfield_path_name(4, id, paths.REGISTRATION_TO_HUNT3))
