"""Step 5: FastSurfer segmentation (docker), transform back to patient space, copy to the patient folder.

Historically sub-step 5.1 ran on a GPU machine with FastSurfer/docker and sub-step
5.2 on a machine with FreeSurfer; the ``.mgz`` / ``.nii.gz`` segmentation folders
were copied between them. Those hand-off folders are the ``segmentation_folder``
arguments; when they are None the patient folder is used.
"""
import os
import shutil

from brainpred import paths
from brainpred.preprocessing.external import run_command
from brainpred.utils import io

# Output name in the data folder -> file produced by FastSurfer in <subject>/mri/
SEG_NAMES_DICT = {paths.INAME_SEGMENTATION_96: "aparc.DKTatlas+aseg.deep.mgz",
                  paths.INAME_SEGMENTATION_36: "aseg.auto_noCCseg.mgz",
                  paths.INAME_FASTSURFER_ORIG: "orig.mgz"}


# ---- generic FastSurfer helpers
def get_fastsurfer_command(image_data_folder, img_name, sid, gpu=True):
    """docker command running FastSurfer on ``<image_data_folder>/<img_name>``, writing to the temp folder."""
    image_output_folder = paths.get_temp_folder()
    license_folder = paths.get_freesurfer_license_folder()
    if gpu:
        # docker run --gpus all -v <data>:/data -v <output>:/output -v <license>:/fs_license --rm --user root deepmi/fastsurfer:latest --fs_license /fs_license/license.txt --t1 /data/<img_name> --sid <subject_id> --sd /output --parallel --seg_only --allow_root --no_cereb --no_biasfield
        return ["docker", "run", "--gpus", "all",
                "-v", f"{image_data_folder}:/data",
                "-v", f"{image_output_folder}:/output",
                "-v", f"{license_folder}:/fs_license",
                "--rm", "--user", "root",
                "deepmi/fastsurfer:latest",
                "--fs_license", "/fs_license/license.txt",
                "--t1", f"/data/{img_name}",
                "--sid", f"{sid}",
                "--sd", "/output",
                "--parallel", "--seg_only",
                "--allow_root",
                "--no_cereb", "--no_biasfield"]
    # docker run -v <data>:/data -v <output>:/output -v <license>:/fs_license --rm --user root deepmi/fastsurfer:cpu-latest --fs_license /fs_license/license.txt --t1 /data/<img_name> --device cpu --sid <subject_id> --sd /output --parallel --seg_only
    return ["docker", "run",
            "-v", f"{image_data_folder}:/data",
            "-v", f"{image_output_folder}:/output",
            "-v", f"{license_folder}:/fs_license",
            "--rm", "--user", "root",
            "deepmi/fastsurfer:cpu-latest",
            "--fs_license", "/fs_license/license.txt",
            "--t1", f"/data/{img_name}",
            "--device", "cpu",
            "--sid", f"{sid}",
            "--sd", "/output",
            "--parallel", "--seg_only",
            "--allow_root",
            "--no_cereb", "--no_biasfield"]


def get_fastsurfer_output_folder(sid):
    return os.path.join(paths.get_temp_folder(), sid, "mri")


def clean_temp_folder():
    temp_folder = paths.get_temp_folder()
    shutil.rmtree(temp_folder, ignore_errors=True)
    os.makedirs(temp_folder, exist_ok=True)


def run_label2vol(seg_mgz_path_name, template_path_name, seg_out_path_name, success_message):
    # mri_label2vol --seg <seg.mgz> --temp <T1 in patient space> --o <seg.nii.gz> --regheader <seg.mgz>
    command = ["mri_label2vol",
               "--seg", seg_mgz_path_name,
               "--temp", template_path_name,
               "--o", seg_out_path_name,
               "--regheader", seg_mgz_path_name]
    return run_command(command, success_message, encoding="utf-8")


# ---- Step 5.1: segmentation (FastSurfer machine)
def get_segmentation_mgz_path_name(hunt, id, name, segmentation_folder=None):
    data_folder = paths.get_patient_folder(hunt, id) if segmentation_folder is None else segmentation_folder
    return os.path.join(data_folder, name.format(id, hunt - 3, "mgz"))


def copy_segmentation_from_temp_to_data(hunt, id, segmentation_folder=None):
    temp_patient_folder = get_fastsurfer_output_folder(id)
    for key, value in SEG_NAMES_DICT.items():
        shutil.copy(os.path.join(temp_patient_folder, value), get_segmentation_mgz_path_name(hunt, id, key, segmentation_folder))


def segment_freesurfer(hunt, id, GPU=False, segmentation_folder=None, verify=True):
    """Run FastSurfer on ``T1_PREP`` and store the ``.mgz`` outputs in ``segmentation_folder``."""
    if verify:
        exist = (io.verify_file(get_segmentation_mgz_path_name(hunt, id, paths.INAME_SEGMENTATION_96, segmentation_folder)) and
                 io.verify_file(get_segmentation_mgz_path_name(hunt, id, paths.INAME_SEGMENTATION_36, segmentation_folder)))
        if exist:
            print(f"FastSurfer segmentation already done for HUNT{hunt} {id}")
            return True

    command = get_fastsurfer_command(paths.get_patient_folder(hunt, id), paths.INAME_PREPROCESSED.format(id, hunt - 3), id, gpu=GPU)
    done = run_command(command, f"Fastsurfer analysis done for HUNT{hunt} {id}", encoding="utf-8")
    if done:
        copy_segmentation_from_temp_to_data(hunt, id, segmentation_folder=segmentation_folder)
        clean_temp_folder()
    return done


# ---- Step 5.2: transform segmentation to patient space (FreeSurfer machine)
def transform_segmentation_to_patient_space(hunt, id, segmentation_mgz_folder=None, segmentation_nii_folder=None):
    patient_folder = paths.get_patient_folder(hunt, id)
    segmentation_mgz_folder = segmentation_mgz_folder if segmentation_mgz_folder is not None else patient_folder
    segmentation_nii_folder = segmentation_nii_folder if segmentation_nii_folder is not None else patient_folder

    for key in SEG_NAMES_DICT:
        run_label2vol(f"{segmentation_mgz_folder}/{key.format(id, hunt - 3, 'mgz')}",
                      f"{patient_folder}/{paths.INAME_REGISTERED_TEMPLATE.format(id, hunt - 3, paths.REGISTRATION_TO_MNI)}",
                      f"{segmentation_nii_folder}/{key.format(id, hunt - 3, 'nii.gz')}",
                      f"Freesurfer transformation SEG done for HUNT{hunt} {id}")


# ---- Step 5.3: copy the .nii.gz segmentations into the patient folder
def copy_seg_to_patient_folder(hunt, id, copy_from):
    patient_folder = paths.get_patient_folder(hunt, id)
    for key in SEG_NAMES_DICT:
        shutil.copy(f"{copy_from}/{key.format(id, hunt - 3, 'nii.gz')}", f"{patient_folder}/{key.format(id, hunt - 3, 'nii.gz')}")
