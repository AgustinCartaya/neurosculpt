
import sys
import os
import shutil

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../'))
sys.path.append(base_path)


import numpy as np
import pydicom
import nibabel as nib
import dicom2nifti # to convert DICOM files to the NIftI format
from pathlib import Path 
import tqdm
import pandas as pd
import utils.functions as fc
import utils.nifti_functions as nfc


# PATH_HUNT_3_AND_4_INFO = f"{current_path}/original_data/GeneralInfo_HUNT_MRI_3_4.xlsx"
# hunt_3_4_df = pd.read_excel(PATH_HUNT_3_AND_4_INFO)
# data_folder = "data_393"
data_folder = "data_missing"
data_path = f"{base_path}/images/{data_folder}"
data_path_sort = f"/images/{data_folder}"

extension_nifti = "nii"
initial_patient_available = "9410000000039"


PATH_HARD_DISK_HUNT4 = f"/media/agustin/Expansion/HUNT-MRI_3&4"
PATH_HARD_DISK_HUNT3 = f"/media/agustin/LaCie"
# names_dict_hunt4 = {"T1": "40002_ORIG_Sag_3D_MPRAGE", "T1_BFC": "2_Sag_3D_MPRAGE"}
names_dict_hunt3 = {"T1": "3", "T1_BFC": "9"}
# names_dict_hunt4 = {"T1_BFC": "2_Sag_3D_MPRAGE"}
# names_dict_hunt3 = {"T1_BFC": "9"}


def get_HUNT_data_folder(hunt, absolute=True):
    if absolute:
        return f"{data_path}/HUNT{hunt}"
    else:
        return f"{data_path_sort}/HUNT{hunt}"


def get_hard_disk_patient_folders(hunt, path_hard_disk):
    # read hard disk elements
    possible_folders = os.listdir(path_hard_disk)

    # obtain hard disk folders
    patient_folders = [e for e in possible_folders if os.path.isdir(os.path.join(path_hard_disk, e))]

    # obtain only folders starting with 9410 (patient folders)
    if hunt == 4:
        patient_folders = [e.split("_")[0] for e in patient_folders if e.startswith("9410")]
    elif hunt == 3:
        patient_folders = [e for e in patient_folders if e.startswith("9410")]

    return patient_folders


## ---------------------- COPY AND CONVERT TO NIFTI HUNT
def copy_individual_hunt_and_convert_to_nifti(hunt, path_hard_disk, names_dict, middle_path=None):
    patient_folders = get_hard_disk_patient_folders(hunt, path_hard_disk)
    patient_folders.sort()
    # print(patient_folders)
    for patient_folder in patient_folders:
        if hunt == 4:
            patient_id = patient_folder.split("_")[0][8:]
        else:
            patient_id = patient_folder[8:]

        original_patient_folder_path = os.path.join(path_hard_disk, patient_folder) 
        patient_folder_path = os.path.join(get_HUNT_data_folder(hunt, True), patient_id) 
        print(patient_folder)
        print(patient_folder_path)
        # verify patient folder (create if do not exists)
        if fc.verify_folder(os.path.join(get_HUNT_data_folder(hunt, False), patient_id), create=True):
            for m_name, m_folder in names_dict.items():
                if middle_path is not None:
                    original_image_folder = os.path.join(original_patient_folder_path, middle_path, m_folder)
                else:
                    original_image_folder = os.path.join(original_patient_folder_path, m_folder) 
                dicom2nifti.convert_directory(original_image_folder, patient_folder_path, compression=False)
                # change image name
                nifti_image_old_name = os.path.join(patient_folder_path, f'{m_folder.lower()}.{extension_nifti}') 
                nifti_image_new_name = os.path.join(patient_folder_path, f'{patient_id}_{hunt-3}_{m_name}.{extension_nifti}') 
                os.rename(nifti_image_old_name, nifti_image_new_name)
        break
        if initial_patient_available.endswith(patient_id):
            break
  
# copy_individual_hunt_and_convert_to_nifti(hunt=4, path_hard_disk=PATH_HARD_DISK_HUNT4, names_dict=names_dict_hunt4)
# copy_individual_hunt_and_convert_to_nifti(hunt=3, path_hard_disk=PATH_HARD_DISK_HUNT3, names_dict=names_dict_hunt3, middle_path="1")


## ---------------------- COPY AND CONVERT TO NIFTI PARALEL HUNT



def copy_parallel_hunt_and_convert_to_nifti():
    """
    Copy and convert to nifti images from both hard disks in parallel.
    a copy list is returned with the following format:
    ID, HUNT3, HUNT4
    ID: patient id
    HUNT3: 1 if copied, 0 if not
    HUNT4: 1 if copied, 0 if not
    """
    copy_list = np.zeros((0, 3), dtype=int)

    # # read hard disk elements
    # possible_folders_HUNT3 = os.listdir(PATH_HARD_DISK_HUNT3)
    # possible_folders_HUNT4 = os.listdir(PATH_HARD_DISK_HUNT4)

    # # obtain hard disk folders
    # patient_folders_HUNT3 = [e for e in possible_folders_HUNT3 if os.path.isdir(os.path.join(PATH_HARD_DISK_HUNT3, e))]
    # patient_folders_HUNT4 = [e for e in possible_folders_HUNT4 if os.path.isdir(os.path.join(PATH_HARD_DISK_HUNT4, e))]

    # # obtain only folders starting with 9410 (patient folders)
    # patient_folders_HUNT3 = [e for e in patient_folders_HUNT3 if e.startswith("9410")]
    # patient_folders_HUNT4 = [e.split("_")[0] for e in patient_folders_HUNT4 if e.startswith("9410")]

    patient_folders_HUNT3 = get_hard_disk_patient_folders(3, PATH_HARD_DISK_HUNT3)
    patient_folders_HUNT4 = get_hard_disk_patient_folders(4, PATH_HARD_DISK_HUNT4)

    # obtain only patients that are in both hard disks
    patient_folders = list(set(patient_folders_HUNT3) & set(patient_folders_HUNT4))

    # select available patients (due to some empty folders)
    patient_folders.sort()
    initial_patient_index = patient_folders.index(initial_patient_available)
    last_patient_index = patient_folders.index(last_patient_available)
    patient_folders = patient_folders[initial_patient_index:last_patient_index+1]

    # copy and convert to nifti
    for _patient_folder in tqdm.tqdm(patient_folders):
        copy_list = np.vstack((copy_list, np.array([int(_patient_folder), -1, -1])))
        # Rest of the code
        for hunt in [3, 4]:
            patient_id = _patient_folder[8:]
            if hunt == 4:
                path_hard_disk = PATH_HARD_DISK_HUNT4
                middle_path = None
                names_dict = names_dict_hunt4
                patient_folder = _patient_folder + "_1"
            else:
                path_hard_disk = PATH_HARD_DISK_HUNT3
                middle_path = "1"
                names_dict = names_dict_hunt3
                patient_folder = _patient_folder

            original_patient_folder_path = os.path.join(path_hard_disk, patient_folder) 
            patient_folder_path = os.path.join(get_HUNT_data_folder(hunt, True), patient_id) 

            # verify patient folder (create if do not exists)
            if fc.verify_folder(os.path.join(get_HUNT_data_folder(hunt, False), patient_id), create=True):
                for m_name, m_folder in names_dict.items():
                    if middle_path is not None:
                        original_image_folder = os.path.join(original_patient_folder_path, middle_path, m_folder)
                    else:
                        original_image_folder = os.path.join(original_patient_folder_path, m_folder) 
                    dicom2nifti.convert_directory(original_image_folder, patient_folder_path, compression=False, reorient=True)
                    # change image name
                    nifti_image_old_name = os.path.join(patient_folder_path, f'{m_folder.lower()}.{extension_nifti}') 
                    nifti_image_new_name = os.path.join(patient_folder_path, f'{patient_id}_{hunt-3}_{m_name}.{extension_nifti}') 
                    try:
                        os.rename(nifti_image_old_name, nifti_image_new_name)
                        copy_list[-1, 1 + hunt-3] = 1
                    except:
                        print(f"ERROR {_patient_folder} HUNT{hunt}")
                        copy_list[-1, 1 + hunt-3] = 0
                    
    return copy_list



# la mayoria de los erores de HUNT3 son por que no existe la carpeta con el bias field corrected (9) o porque la carpeta existe y no tiene imagenes
# la mayoria de los errores de HUNT4 son porque la carpeta 2_Sag_3D_MPRAGE comienza por 4 o 5, es decir 4_Sag_3D_MPRAGE o 5_Sag_3D_MPRAGE o cualquier otro numero
def removed_not_copied_images_folder_and_pairs():
    """
    Remove not copied images and folders.
    the copy list is read and the patiets that do not exists in both HUNT3 and HUNT4 are removed.
    """
    copy_list = pd.read_csv(f'{data_path}/copy_list.csv')
    ids_to_remove = copy_list[(copy_list["HUNT3"] == 0) | (copy_list["HUNT4"] == 0)]["ID"].values
    folders_to_remove = [str(e)[8:] for e in ids_to_remove if str(e).startswith("9410")]
    for patient_folder in folders_to_remove:
        try:
            shutil.rmtree(os.path.join(get_HUNT_data_folder(3, True), patient_folder))
            shutil.rmtree(os.path.join(get_HUNT_data_folder(4, True), patient_folder))
            print(f"Removed {patient_folder}")
        except:
            print(f"Already removed {patient_folder}")


def copy_data_from_hard_disk():
    """
    Perform the copy and and the removal of not copied images and folders.
    """
    copy_list = copy_parallel_hunt_and_convert_to_nifti()
    df = pd.DataFrame(copy_list, columns=["ID", "HUNT3", "HUNT4"])
    df.to_csv(f'{data_path}/copy_list.csv', index=False)

    removed_not_copied_images_folder_and_pairs()



## ---------------------- FIND COPIED IMAGES WITH DIFFERENT DIMENSIONS
def find_TI_BFC_in_different_folder_HUNT3(patient_id):
    try:
        
        original_patient_folder_path = os.path.join(PATH_HARD_DISK_HUNT3, f"94100000{patient_id}","1") 
        possible_folders = os.listdir(original_patient_folder_path)
        possible_folders = [int(e) for e in possible_folders if os.path.isdir(os.path.join(original_patient_folder_path, e))]
        possible_folders.sort()

        initial_folder = possible_folders.index(9)
        initial_folder = possible_folders.index(9)
        possible_folders = possible_folders[initial_folder:]

        for folder in possible_folders:
            dicom_folder = os.path.join(original_patient_folder_path, str(folder))
            dicom_files = [file for file in os.listdir(dicom_folder) if file.endswith('.dcm')]
            count = len(dicom_files)
            if 160 < count < 170:
                return folder
    except:
        pass
    return None

def find_images_with_different_dimensions():
    copy_list = pd.read_csv(f'{data_path}/copy_list_new.csv')
    np_list = copy_list.values
    hunt = 3
    c=0

    for i, row_list in enumerate(np_list):
        if row_list[1] == 1 and row_list[2] == 1:
            patient_folder = str(row_list[0])
            patient_id = patient_folder[8:]
         
            img, affine = nfc.load_nifti(f"{get_HUNT_data_folder(hunt, True)}/{patient_id}/{patient_id}_{hunt-3}_T1_BFC.nii")
            img_size = img.shape
            if img_size[-1] < 160:
                np_list[i, 1] = 0
                c+=1

                # different_folder = find_TI_BFC_in_different_folder_HUNT3(patient_id)
                # if different_folder is None:
                #     print(f"{patient_id} HUNT{hunt} {img_size}")
                # else:
                #     print(f"{patient_id} HUNT{hunt} {img_size} -> {different_folder}")

                # 11,
    df = pd.DataFrame(copy_list, columns=["ID", "HUNT3", "HUNT4"])
    df.to_csv(f'{data_path}/copy_list_new2.csv', index=False)
      
    print(c)
# find_images_with_different_dimensions()
    
# 33
## ---------------------- FIND IMAGES WITH DIFFERENT DIMENSIONS IN HARD DISK

def copy_and_convert_to_nifty_incomplete_data(_patient_folder, new_dicom_folder):

    # Rest of the code
    for hunt in [3, 4]:
        patient_id = _patient_folder[8:]
        if hunt == 4:
            path_hard_disk = PATH_HARD_DISK_HUNT4
            middle_path = None
            names_dict = names_dict_hunt4
            patient_folder = _patient_folder + "_1"
        else:
            path_hard_disk = PATH_HARD_DISK_HUNT3
            middle_path = "1"
            # names_dict = names_dict_hunt3
            names_dict = {"T1_BFC": str(new_dicom_folder)}

            patient_folder = _patient_folder

        original_patient_folder_path = os.path.join(path_hard_disk, patient_folder) 
        patient_folder_path = os.path.join(get_HUNT_data_folder(hunt, True), patient_id) 

        # verify patient folder (create if do not exists)
        fc.create_folders(os.path.join(get_HUNT_data_folder(hunt, False), patient_id))
        if fc.verify_folder(os.path.join(get_HUNT_data_folder(hunt, True), patient_id), create=False):
            for m_name, m_folder in names_dict.items():
                if middle_path is not None:
                    original_image_folder = os.path.join(original_patient_folder_path, middle_path, m_folder)
                else:
                    original_image_folder = os.path.join(original_patient_folder_path, m_folder) 
                dicom2nifti.convert_directory(original_image_folder, patient_folder_path, compression=False, reorient=True)
                # change image name
                nifti_image_old_name = os.path.join(patient_folder_path, f'{m_folder.lower()}.{extension_nifti}') 
                nifti_image_new_name = os.path.join(patient_folder_path, f'{patient_id}_{hunt-3}_{m_name}.{extension_nifti}') 
                try:
                    os.rename(nifti_image_old_name, nifti_image_new_name)
                except:
                    print(f"ERROR {_patient_folder} HUNT{hunt}")
                    return False              
    return True


def find_images_with_different_dimensions_hard_disk():
    copy_list = pd.read_csv(f'{data_path}/copy_list.csv')
    np_list = copy_list.values
    c=0
    
    for i, row_list in enumerate(copy_list.values):
        if row_list[1] == 0 and row_list[2] == 1:
            patient_folder = str(row_list[0])
            different_folder = find_TI_BFC_in_different_folder_HUNT3(patient_folder[8:])
            if different_folder is None:
                print(f"{patient_folder[8:]} HUNT{3}")
                c+=1
            else:
                copied = copy_and_convert_to_nifty_incomplete_data(patient_folder, different_folder)
                if copied:
                    np_list[i, 1] = 1
                print(f"{patient_folder[8:]} HUNT{3} -> {different_folder} = {np_list[i, 1]}")

    df = pd.DataFrame(copy_list, columns=["ID", "HUNT3", "HUNT4"])
    df.to_csv(f'{data_path}/copy_list_new.csv', index=False)
  

    print(c)

# find_images_with_different_dimensions_hard_disk()
