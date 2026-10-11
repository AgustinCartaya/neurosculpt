
import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../'))
sys.path.append(base_path)

import numpy as np
import subprocess
import project_images_loader as pil
import shutil
import utils.nifti_functions as nfc
import utils.functions as fc
import utils.util as util


## --------------- STEP 1: SEGMENTATION ---------------         
def get_degmentation_path_name(hunt, id, name, segmentation_folder):
    if  segmentation_folder is None:
        data_folder = pil.get_patient_folder(hunt, id, absolute=True)
    else:
        data_folder = segmentation_folder

    return os.path.join(data_folder, name.format(id, hunt-3, "mgz"))


def copy_segmentation_from_temp_to_data(id, path_name_out):
    temp_patient_folder = os.path.join(pil.get_temp_folder(absolute=True), id, "mri") 
    if path_name_out.endswith("nii.gz"):
        path_name_out = path_name_out.replace("nii.gz", "mgz")
    shutil.copy(os.path.join(temp_patient_folder, "aparc.DKTatlas+aseg.deep.mgz"), path_name_out)


def clean_temp_folder():
    temp_folder = pil.get_temp_folder(absolute=True)
    shutil.rmtree(temp_folder, ignore_errors=True)
    os.makedirs(temp_folder, exist_ok=True)


def segment_freesurfer(id, path_name_in, path_name_out, GPU=False):
    # image_data_folder = f"{current_path}/images/to_copy"
    image_data_folder = os.path.dirname(path_name_in)
    image_output_folder = f"{pil.get_temp_folder(absolute=True)}"
    license_folder = f"{pil.get_freesurfer_license_folder(absolute=True)}"

    # docker run --gpus all -v <path_folder_data>:/data -v <path_folder_output>:/output -v <license>:/fs_license --rm --user root deepmi/fastsurfer:latest --fs_license /fs_license/license.txt --t1 /data/<img_name> --sid <subjet_id> --sd /output --parallel --seg_only --allow_root --no_cereb --no_biasfield
    if GPU:
        command = ["docker", "run", "--gpus", "all",
               "-v", f"{image_data_folder}:/data", 
               "-v", f"{image_output_folder}:/output", 
               "-v", f"{license_folder}:/fs_license",
               "--rm", "--user", "root",
               "deepmi/fastsurfer:latest",
               "--fs_license", "/fs_license/license.txt", 
                "--t1", f"/data/{os.path.basename(path_name_in)}",
                "--sid", f"{id}",
                "--sd", "/output",
                "--parallel", "--seg_only",
                "--allow_root",
                "--no_cereb", "--no_biasfield"]
    # docker run -v <path_folder_data>:/data -v <path_folder_output>:/output -v <license>:/fs_license --rm --user root deepmi/fastsurfer:cpu-latest --fs_license /fs_license/license.txt --t1 /data/<img_name> --device cpu --sid <subjet_id> --sd /output --parallel --seg_only
    else:
        command = ["docker", "run", 
                "-v", f"{image_data_folder}:/data", 
                "-v", f"{image_output_folder}:/output", 
                "-v", f"{license_folder}:/fs_license",
                "--rm", "--user", "root",
                "deepmi/fastsurfer:cpu-latest",
                "--fs_license", "/fs_license/license.txt", 
                "--t1", f"/data/{os.path.basename(path_name_in)}",
                "--device", "cpu",
                "--sid", f"{id}",
                "--sd", "/output",
                "--parallel", "--seg_only", 
                "--allow_root",
                "--no_cereb", "--no_biasfield"]
    
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')

    if result.returncode == 0:
        copy_segmentation_from_temp_to_data(id, path_name_out)
        clean_temp_folder()
        print(f"Fastsurfer analysis done for HUNT{3} {id}")
    else:
        print("Error:", result.stderr)



## --------------- STEP 2: TRANSFORM SEGMENTATION TO PATIENT SPACE ---------------
def transform_segmentation_to_patient_space(id, path_name, t1_img_path_name):
    # mri_label2vol --seg 00039_0_SEG.mgz --temp 00039_0_T1_PREP.nii --o aseg-in-rawavg2.mgz --regheader 00039_0_SEG.mgz
    path_name_in = path_name.replace("nii.gz", "mgz")
    command = ["mri_label2vol", 
                "--seg", f"{path_name_in}",
                "--temp", f"{t1_img_path_name}",
                "--o", f"{path_name}",
                "--regheader", f"{path_name_in}"]
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')

    if result.returncode == 0:
        print(f"Freesurfer transformation SEG done for HUNT{3} {id}")
    else:
        print("Error:", result.stderr)


# def copy_seg_to_patient_folder(hunt, id, copy_from):
#     patient_folder = f"{pil.get_patient_folder(hunt, id, absolute=True)}"
#     for key, value in seg_names_dict.items():
#         shutil.copy(f"{copy_from}/{key.format(id, hunt-3, 'nii.gz')}", f"{patient_folder}/{key.format(id, hunt-3, 'nii.gz')}")


