
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

seg_names_dict = {pil.INAME_SEGMENTATION_96 : "aparc.DKTatlas+aseg.deep.mgz", 
                  pil.INAME_SEGMENTATION_36 : "aseg.auto_noCCseg.mgz",
                  pil.INAME_FASTSURFER_ORIG : "orig.mgz"
                  }

## --------------- STEP 1: SEGMENTATION ---------------
def get_degmentation_path_name(hunt, id, name, segmentation_folder):
    if  segmentation_folder is None:
        data_folder = pil.get_patient_folder(hunt, id, absolute=True)
    else:
        data_folder = segmentation_folder

    return os.path.join(data_folder, name.format(id, hunt-3, "mgz"))
                 
def copy_segmentation_from_temp_to_data(hunt, id, segmentation_folder=None):
    temp_patient_folder = os.path.join(pil.get_temp_folder(absolute=True), id, "mri") 
    for key, value in seg_names_dict.items():
        shutil.copy(os.path.join(temp_patient_folder, value), get_degmentation_path_name(hunt, id, key, segmentation_folder))


def clean_temp_folder():
    temp_folder = pil.get_temp_folder(absolute=True)
    shutil.rmtree(temp_folder, ignore_errors=True)
    os.makedirs(temp_folder, exist_ok=True)


def segment_freesurfer(hunt, id, GPU=False, segmentation_folder=None, verify=True):
    if verify:
        exist = (fc.verify_file(get_degmentation_path_name(hunt, id, pil.INAME_SEGMENTATION_96, segmentation_folder)) and 
                 fc.verify_file(get_degmentation_path_name(hunt, id, pil.INAME_SEGMENTATION_36, segmentation_folder)))
        if exist:
            print(f"SynthSR already done for HUNT{hunt} {id}")
            return True
        
    # image_data_folder = f"{current_path}/images/to_copy"
    image_data_folder = f"{pil.get_patient_folder(hunt, id, absolute=True)}"
    image_output_folder = f"{pil.get_temp_folder(absolute=True)}"
    license_folder = f"{pil.get_freesurfer_license_folder(absolute=True)}"
    img_name = pil.INAME_PREPROCESSED.format(id, hunt-3)

    # docker run --gpus all -v <path_folder_data>:/data -v <path_folder_output>:/output -v <license>:/fs_license --rm --user root deepmi/fastsurfer:latest --fs_license /fs_license/license.txt --t1 /data/<img_name> --sid <subjet_id> --sd /output --parallel --seg_only --allow_root --no_cereb --no_biasfield
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
    # docker run -v <path_folder_data>:/data -v <path_folder_output>:/output -v <license>:/fs_license --rm --user root deepmi/fastsurfer:cpu-latest --fs_license /fs_license/license.txt --t1 /data/<img_name> --device cpu --sid <subjet_id> --sd /output --parallel --seg_only
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
    
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')

    if result.returncode == 0:
        copy_segmentation_from_temp_to_data(hunt, id, segmentation_folder=segmentation_folder)
        clean_temp_folder()
        print(f"Fastsurfer analysis done for HUNT{hunt} {id}")
    else:
        print("Error:", result.stderr)



## --------------- STEP 2: TRANSFORM SEGMENTATION TO PATIENT SPACE ---------------
def transform_segmentation_to_patient_space(hunt, id, segmentation_mgz_folder=None, segmentation_nii_folder=None):
    # mri_label2vol --seg 00039_0_SEG.mgz --temp 00039_0_T1_PREP.nii --o aseg-in-rawavg2.mgz --regheader 00039_0_SEG.mgz
    
    patient_folder = f"{pil.get_patient_folder(hunt, id, absolute=True)}"
    segmentation_mgz_folder = segmentation_mgz_folder if segmentation_mgz_folder is not None else patient_folder
    segmentation_nii_folder = segmentation_nii_folder if segmentation_nii_folder is not None else patient_folder


    for key, value in seg_names_dict.items():
        command = ["mri_label2vol", 
                   "--seg", f"{segmentation_mgz_folder}/{key.format(id, hunt-3, 'mgz')}",
                   "--temp", f"{patient_folder}/{pil.INAME_REGISTERED_TEMPLATE.format(id, hunt-3, pil.REGISTRATION_TO_MNI)}",
                   "--o", f"{segmentation_nii_folder}/{key.format(id, hunt-3, 'nii.gz')}",
                   "--regheader", f"{segmentation_mgz_folder}/{key.format(id, hunt-3, 'mgz')}"]
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')

        if result.returncode == 0:
            print(f"Freesurfer transformation SEG done for HUNT{hunt} {id}")
        else:
            print("Error:", result.stderr)


def copy_seg_to_patient_folder(hunt, id, copy_from):
    patient_folder = f"{pil.get_patient_folder(hunt, id, absolute=True)}"
    for key, value in seg_names_dict.items():
        shutil.copy(f"{copy_from}/{key.format(id, hunt-3, 'nii.gz')}", f"{patient_folder}/{key.format(id, hunt-3, 'nii.gz')}")



# ----------------- NOT USED: CHANGE SEGMENTATION NAMES -----------------
def change_segmentation_names():
    matching_ids = pil.get_matching_ids()
    # for id in matching_ids[1:]:
        # for hunt in [3, 4]:

            # # patient folders
            # patient_path = pil.get_patient_folder(hunt, id, absolute=True)
            # os.rename(os.path.join(patient_path, f"{id}_{hunt-3}_SEG.nii.gz"), os.path.join(patient_path, f"{id}_{hunt-3}_SEG_96.nii.gz"))
            # os.rename(os.path.join(patient_path, f"{id}_{hunt-3}_SEG_SIMPLE.nii.gz"), os.path.join(patient_path, f"{id}_{hunt-3}_SEG_36.nii.gz"))


            # # Segmentations
            # segmentation_folder_mgz = f"C:/Users/Agustin/Master/Master/S4/code/images/segmentations/segmentation_mgz"
            # os.rename(os.path.join(segmentation_folder_mgz, f"{id}_{hunt-3}_SEG.mgz"), os.path.join(segmentation_folder_mgz, f"{id}_{hunt-3}_SEG_96.mgz"))
            # os.rename(os.path.join(segmentation_folder_mgz, f"{id}_{hunt-3}_SEG_SIMPLE.mgz"), os.path.join(segmentation_folder_mgz, f"{id}_{hunt-3}_SEG_36.mgz"))

            # segmentation_folder_nii = f"C:/Users/Agustin/Master/Master/S4/code/images/segmentations/segmentation_nii"
            # os.rename(os.path.join(segmentation_folder_nii, f"{id}_{hunt-3}_SEG.nii.gz"), os.path.join(segmentation_folder_nii, f"{id}_{hunt-3}_SEG_96.nii.gz"))
            # os.rename(os.path.join(segmentation_folder_nii, f"{id}_{hunt-3}_SEG_SIMPLE.nii.gz"), os.path.join(segmentation_folder_nii, f"{id}_{hunt-3}_SEG_36.nii.gz"))


        # # registrations
        # registrations_elastix_path = f"{base_path}/test_code/dfield_approach_elastix/images/registrations"
        # os.rename(os.path.join(registrations_elastix_path, f"{id}_0_SEG_0038.nii.gz"), os.path.join(registrations_elastix_path, f"{id}_0_SEG_96_0038.nii.gz"))
        # os.rename(os.path.join(registrations_elastix_path, f"{id}_0_SEG_SIMPLE_0038.nii.gz"), os.path.join(registrations_elastix_path, f"{id}_0_SEG_36_0038.nii.gz"))


    # joint segmentation NOT USED ANY MORE
    # segmentation_folder_mgz_missing = "C:/Users/Agustin/Master/Master/S4/code/images/segmentations/segmentation_data_missing_nii"
    # segmentation_folder_mgz_393 = "C:/Users/Agustin/Master/Master/S4/code/images/segmentations/segmentation_data_393_nii/HUNT{0}/{1}"
    # for id in matching_ids:
    #     for hunt in [3, 4]:
    #         if fc.verify_folder(segmentation_folder_mgz_393.format(hunt, id)):
    #             for archivo in os.listdir(segmentation_folder_mgz_393.format(hunt, id)):
    #                 ruta_archivo_origen = f"{segmentation_folder_mgz_393.format(hunt, id)}/{archivo}"
    #                 ruta_archivo_destino = f"{segmentation_folder_mgz_missing}/{archivo}"
    #                 # os.chmod(segmentation_folder_mgz_393.format(hunt), 0o777) 
    #                 shutil.copy2(ruta_archivo_origen, ruta_archivo_destino)

# for id in pil.test_patients_ids[:2]:
#     for hunt in pil.HUNTS:
#         print(f"FastSurfer segmentation for HUNT{hunt} {id}")
#     # save_skull_stripping_freesurfer(id, "0038")
#         segment_freesurfer(hunt, id, GPU=True)

# segment_freesurfer(3, "00039", GPU=True)
# copy_segmentation_from_temp_to_data(3, "00039")
# clean_temp_folder()
# print(pil.get_freesurfer_license_folder(absolute=True))