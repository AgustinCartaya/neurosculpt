
import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(base_path)

import numpy as np
import subprocess
import project_images_loader as pil
import utils.nifti_functions as nfc
import utils.functions as fc
import utils.util as util

import license.key as key

def save_skull_stripping_freesurfer(id, name):
    image_data_folder = f"{current_path}/images/data"

    #mri_synthstrip -i 12082_0_T1_SYNTHSR.nii -o stripped2.nii -m mask.mgz --no-csf
    img_path_name = f"{image_data_folder}/{id}_{0}_T1_{name}.nii"
    mask_path_name = f"{image_data_folder}/{id}_{0}_BMASK_{name}.nii"
    img_temp_path_name = f"{image_data_folder}/temp.nii"

    command = ["mri_synthstrip", "-i", img_path_name, "-o", img_temp_path_name, "--no-csf", "-m", mask_path_name]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        os.remove(img_temp_path_name)
        print(f"Freesurfer skull stripping done for HUNT{3} {id}")
    else:
        print("Error:", result.stderr)


def evaluate_registrqtion_with_reql_skull_stripping():     
    image_data_folder = f"{current_path}/images/data"

    id = 12082
    mask_HUNT3_path_name = f"{image_data_folder}/{id}_{0}_BMASK_{'0038'}.nii"
    mask_HUNT4_path_name = f"{image_data_folder}/{id}_{1}_BMASK_{'MNI'}.nii"

    img_HUNT3_path_name = f"{image_data_folder}/{id}_{0}_T1_{'0038'}.nii"
    img_HUNT4_path_name = f"{image_data_folder}/{id}_{1}_T1_{'MNI'}.nii"


    mask_HUNT3, _ = nfc.load_nifti(mask_HUNT3_path_name) 
    mask_HUNT4, _ = nfc.load_nifti(mask_HUNT4_path_name) 

    img_HUNT3, _ = nfc.load_nifti(img_HUNT3_path_name) 
    img_HUNT4, _ = nfc.load_nifti(img_HUNT4_path_name) 

    print(np.sum(mask_HUNT3-mask_HUNT4))
    print(util.get_metrics_dict(util.skull_stripping_and_normalize(img_HUNT3, mask_HUNT3),
                                util.skull_stripping_and_normalize(img_HUNT4, mask_HUNT4)))
    fc.imgshow_3D("", mask_HUNT3 - mask_HUNT4)


def segment_freesurfer(hunt, id, GPU=False):
    # docker run -v <path_folder_data>:/data -v <path_folder_output>:/output -v <license>:/fs_license --rm --user root deepmi/fastsurfer:cpu-latest --fs_license /fs_license/license.txt --t1 /data/<img_name> --device cpu --sid <subjet_id> --sd /output --parallel --seg_only
    # docker run --gpus all -v <path_folder_data>:/data -v <path_folder_output>:/output -v <license>:/fs_license --rm --user root deepmi/fastsurfer:latest --fs_license /fs_license/license.txt --t1 /data/<img_name> --sid <subjet_id> --sd /output --parallel --seg_only --allow_root --no_cereb --no_biasfield
    
    
    # image_data_folder = f"{current_path}/images/to_copy"
    image_data_folder = f"{pil.get_patient_folder(hunt, id, absolute=True)}"
    image_output_folder = f"{current_path}/images/output_preprocessed/HUNT{hunt}"
    license_folder = f"{current_path}/license"
    img_name = f"{id}_{hunt-3}_T1_PREP.nii"

    if GPU:
        command = ["docker", "run", "--gpus", "all",
               "-v", f"{image_data_folder}:/data", 
               "-v", f"{image_output_folder}:/output", 
               "-v", f"{license_folder}:/fs_license",
               "--rm", "--user", "root",
               "deepmi/fastsurfer:latest",
               "--fs_license", "/fs_license/license.txt", 
                "--t1", f"/data/{img_name}",
                "--sid", f"{id}",
                "--sd", "/output",
                "--parallel", "--seg_only",
                "--allow_root",
                "--no_cereb", "--no_biasfield"]
    else:
        command = ["docker", "run", 
                "-v", f"{image_data_folder}:/data", 
                "-v", f"{image_output_folder}:/output", 
                "-v", f"{license_folder}:/fs_license",
                "--rm", "--user", "root",
                "deepmi/fastsurfer:cpu-latest",
                "--fs_license", "/fs_license/license.txt", 
                    "--t1", f"/data/{img_name}",
                    "--device", "cpu",
                    "--sid", f"{id}",
                    "--sd", "/output",
                    "--parallel", "--seg_only", 
                    "--allow_root",
                    "--no_cereb", "--no_biasfield"]
    
    # command_str = ' '.join(command)
    # print(command_str)


    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')

    if result.returncode == 0:
        print(f"Fastsurfer analysis done for HUNT{3} {id}")
    else:
        print("Error:", result.stderr)

for id in pil.test_patients_ids[:2]:
    for hunt in pil.HUNTS:
        print(f"FastSurfer segmentation for HUNT{hunt} {id}")
    # save_skull_stripping_freesurfer(id, "0038")
        segment_freesurfer(hunt, id, GPU=True)
