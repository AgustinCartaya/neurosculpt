
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
from tqdm import tqdm
import pandas as pd
import utils.functions as fc
import utils.nifti_functions as nfc




DATA_FOLDER = "data_missing"
DATA_PATH = f"{base_path}/images/{DATA_FOLDER}"
DATA_PATH_SHORT = f"images/{DATA_FOLDER}"
PATH_COPY_LIST_393 = f"{DATA_PATH}/copy_list_393.csv"
PATH_COPY_LIST = f"{DATA_PATH}/copy_list.csv"
PATH_HARD_DISK_HUNT3 = "/media/agustin/LaCie"

INAME = "{0}_{1}_T1_{2}.nii"

PATH_HARD_DISK_HUNT4 = ["/media/agustin/LaCie1/HUNT4_bok1to8/Forsendelse1_NT4MRI2 2018-04-20",
                         "/media/agustin/LaCie1/HUNT4_bok1to8/Forsendelse2_NT4MRI2 2018-05-02",
                         "/media/agustin/LaCie1/HUNT4_bok1to8/Forsendelse3_NT4MRI2 2018-06-27",
                         "/media/agustin/LaCie1/HUNT4_bok1to8/Forsendelse4_NT4MRI2 2018-06-28",
                         "/media/agustin/LaCie1/HUNT4_bok1to8/Forsendelse5_NT4MRI2 2018-07-06",
                         "/media/agustin/LaCie1/HUNT4_bok1to8/Forsendelse6_NT4MRI2 2018-08-22",
                         "/media/agustin/LaCie1/HUNT4_bok1to8/Forsendelse7_NT4MRI2 2018-09-11",
                         "/media/agustin/LaCie1/HUNT4_bok1to8/Forsendelse8_NT4MRI2 2018-10-04",
                         
                         "/media/agustin/LaCie2/anon9to22_hunt3-4/Forsendelse9_NT4MRI2 2018-10-28",
                         "/media/agustin/LaCie2/anon9to22_hunt3-4/Forsendelse10_NT4MRI2 2018-11-16",
                         "/media/agustin/LaCie2/anon9to22_hunt3-4/Forsendelse11_NT4MRI2 2018-12-12",
                         "/media/agustin/LaCie2/anon9to22_hunt3-4/Forsendelse12_NT4MRI2 2019-02-01",
                         "/media/agustin/LaCie2/anon9to22_hunt3-4/Forsendelse13_NT4MRI2 2019-02-18",
                         "/media/agustin/LaCie2/anon9to22_hunt3-4/Forsendelse14_NT4MRI2 2019-03-21",
                         "/media/agustin/LaCie2/anon9to22_hunt3-4/Forsendelse15_NT4MRI2 2019-04-10",
                         "/media/agustin/LaCie2/anon9to22_hunt3-4/Forsendelse16_NT4MRI2 2019-07-05",
                         "/media/agustin/LaCie2/anon9to22_hunt3-4/Forsendelse17_NT4MRI2 2019-09-19",
                         "/media/agustin/LaCie2/anon9to22_hunt3-4/Forsendelse18_NT4MRI2 2019-10-04",
                         "/media/agustin/LaCie2/anon9to22_hunt3-4/Forsendelse19_NT4MRI2 2019-10-09",
                         "/media/agustin/LaCie2/anon9to22_hunt3-4/Forsendelse20_NT4MRI2 2019-11-01",
                         "/media/agustin/LaCie2/anon9to22_hunt3-4/Forsendelse21_NT4MRI2 2019-11-25",
                         "/media/agustin/LaCie2/anon9to22_hunt3-4/Forsendelse22_NT4MRI2 2019-11-25",
                         "/media/agustin/LaCie2/HUNT3_4_LongNum_missing"
                         ]


def get_local_patient_data_folder(hunt, patient_id, absolute=True):
    if absolute:
        return f"{DATA_PATH}/HUNT{hunt}/{patient_id}"
    else:
        return f"{DATA_PATH_SHORT}/HUNT{hunt}/{patient_id}"


class T1wImagePathSearcher:
    DICOM_ATT_FILTERED = "ScanOptions"

    HUNT3_T1W_RAW = ['FAST_GEMS', 'EDR_GEMS']
    HUNT3_T1W_BFC = ['FAST_GEMS', 'EDR_GEMS', 'FILTERED_GEMS']

    HUNT4_T1W_RAW = ['EDR_GEMS', 'ACC_GEMS', 'IR_GEMS']
    HUNT4_T1W_BFC = ['EDR_GEMS', 'FILTERED_GEMS', 'ACC_GEMS', 'IR_GEMS']

    MIN_HUNT3_LAYERS = 160
    MIN_HUNT4_LAYERS = 200

    """find the path of the T1w RAW and BFC (if exists) image in the patient folder"""
    def __init__(self, patient_folder, hunt):
        self.patient_folder = patient_folder
        self.folder_raw = None
        self.folder_bfc = None
        self.hunt = hunt

        if self.hunt == 3:
            self.t1W_raw = self.HUNT3_T1W_RAW
            self.t1W_bfc = self.HUNT3_T1W_BFC
            self.min_layers = self.MIN_HUNT3_LAYERS
        elif self.hunt == 4:
            self.t1W_raw = self.HUNT4_T1W_RAW
            self.t1W_bfc = self.HUNT4_T1W_BFC
            self.min_layers = self.MIN_HUNT4_LAYERS


        self.__find_dicom_folders()

    def __find_dicom_folders(self):
        def same_elements(list1, list2):
            return set(list1) == set(list2)

        for root, dirs, files in os.walk(self.patient_folder):
            try:
                if len(files) > self.min_layers:
                    dicom_files = [file for file in files if file.endswith('.dcm')]
                    dicom = pydicom.dcmread(os.path.join(root, dicom_files[0]), stop_before_pixels=True)
                    dicom_attr = dicom["ScanOptions"]
                    dicom_values = list(dicom_attr.value)
                    if same_elements(dicom_values, self.t1W_raw):
                        self.folder_raw = root
                    elif same_elements(dicom_values, self.t1W_bfc):
                        self.folder_bfc = root
                    
                    if self.folder_raw is not None and self.folder_bfc is not None:
                        break
            except:
                pass                    



class PatientFolderPathSearcher:
    """find the path of the patient folder"""
    def __init__(self, base_folder_list, patient_folder_name, hunt):
        self.base_folder_list = base_folder_list
        if hunt == 4:
            self.patient_folder_name = f"{patient_folder_name}_1"
        else:
            self.patient_folder_name = patient_folder_name
        self.hunt = hunt
        self.patient_folder_path = self.__find_patient_folder()


    def __find_patient_folder(self):
        """
        Search for a folder with the name patient_folder_name in the given base_folder.
        """
        for base_folder in self.base_folder_list:
            for folder in os.listdir(base_folder):
                if folder == self.patient_folder_name:
                    return os.path.join(base_folder, folder)


    def __deep_find_patient_folder(self):
        """
        Search for a folder with the name patient_folder_name in the given base_folder.
        It may be contained in one or more subfolders of base_folder.
        Returns the path of the folder if found, None otherwise.
        """
        for base_folder in self.base_folder_list:
            for root, dirs, files in os.walk(base_folder):
                if self.patient_folder_name in dirs:
                    return os.path.join(root, self.patient_folder_name)
        return None



def get_HUNT3_patient_folders(base_folder):
    # read hard disk elements
    possible_folders = os.listdir(base_folder)
    # obtain hard disk folders
    patient_folders = [e for e in possible_folders if os.path.isdir(os.path.join(base_folder, e))]
    # obtain only folders starting with 9410 (patient folders)
    patient_folders = [e for e in patient_folders if e.startswith("9410")]
    return patient_folders


def remove_copied_folders(copy_list_path_name, patient_folders):
    copy_list = pd.read_csv(copy_list_path_name)
    folders_copied = copy_list[(copy_list["HUNT3"] == 1) & (copy_list["HUNT4"] == 1)]["ID"].values
    folders_copied = [str(folder) for folder in folders_copied]
    patient_folders = list(set(patient_folders) - set(folders_copied))
    return patient_folders


def find_images_folders_to_copy():
    patient_folder_list = get_HUNT3_patient_folders(PATH_HARD_DISK_HUNT3)
    print(f"Total patient folders: {len(patient_folder_list)}")
    patient_folder_list = remove_copied_folders(PATH_COPY_LIST_393, patient_folder_list)
    print(f"Patient folders to copy: {len(patient_folder_list)}")
    patient_folder_list.sort()

    counter_available_folders = 0
    counter_available_raw = 0
    counter_available_bfc = 0
    available_images = []

    for patient_folder in tqdm(patient_folder_list):
        pfr_hunt3 = PatientFolderPathSearcher([PATH_HARD_DISK_HUNT3], patient_folder, 3)
        pfr_hunt4 = PatientFolderPathSearcher(PATH_HARD_DISK_HUNT4, patient_folder, 4)

        if pfr_hunt3.patient_folder_path is not None and pfr_hunt4.patient_folder_path is not None:
            ipr_hunt3 = T1wImagePathSearcher(pfr_hunt3.patient_folder_path, 3)
            ipr_hunt4 = T1wImagePathSearcher(pfr_hunt4.patient_folder_path, 4)

            if ipr_hunt3.folder_bfc is not None and ipr_hunt4.folder_bfc is not None:
                available_images.append((patient_folder, "BFC", ipr_hunt3.folder_bfc, ipr_hunt4.folder_bfc))
                counter_available_bfc += 1
            elif ipr_hunt3.folder_raw is not None and ipr_hunt4.folder_raw is not None:
                available_images.append((patient_folder, "RAW", ipr_hunt3.folder_raw, ipr_hunt4.folder_raw))
                counter_available_raw += 1
            else:
                print(f"Missing images for {patient_folder}")
            counter_available_folders += 1 


    print(f"Total patient folders : {counter_available_folders}")
    print(f"Total bfc : {counter_available_bfc}")
    print(f"Total raw : {counter_available_raw}")

    return available_images

def convert_dicom_to_nifti(dicom_folder, nifti_folder, nifti_file_name):
    dicom2nifti.convert_directory(dicom_folder, nifti_folder, compression=False, reorient=True)
    nifti_old_name = os.path.basename(dicom_folder).lower() + ".nii"
    try:
        os.rename(os.path.join(nifti_folder, nifti_old_name ), os.path.join(nifti_folder, nifti_file_name))
        return True
    except:
        return False

def copy_images():
    available_images = find_images_folders_to_copy()
    copy_list = []
    # verify patient folder (create if do not exists)
    for patient_folder, image_type, folder_hunt3, folder_hunt4 in tqdm(available_images):
        patient_id = patient_folder[8:]
        patient_folder_path_hunt3 = get_local_patient_data_folder(3, patient_id, True)
        patient_folder_path_hunt4 = get_local_patient_data_folder(4, patient_id, True)

        img_name_hunt3 = INAME.format(patient_id, 0, image_type)
        img_name_hunt4 = INAME.format(patient_id, 1, image_type)

        conv_hunt3 = False
        conv_hunt4 = False

        if (fc.verify_folder(get_local_patient_data_folder(3, patient_id, False), create=True) and 
            fc.verify_folder(get_local_patient_data_folder(4, patient_id, False), create=True)):

            copy_list.append([patient_folder, 0, 0])

            conv_hunt3 = convert_dicom_to_nifti(folder_hunt3, patient_folder_path_hunt3, img_name_hunt3)
            if conv_hunt3:
                copy_list[-1][1] = 1
                conv_hunt4 = convert_dicom_to_nifti(folder_hunt4, patient_folder_path_hunt4, img_name_hunt4)
                if conv_hunt4:
                    copy_list[-1][2] = 1

            if not (conv_hunt3 and conv_hunt4):
                print(f"Error copying {patient_id} {image_type} removing folders")
                for folder_path in (patient_folder_path_hunt3, patient_folder_path_hunt4):
                    try:
                        shutil.rmtree(folder_path)
                    except:
                        pass

       
    df = pd.DataFrame(copy_list, columns=["ID", "HUNT3", "HUNT4"])
    df.to_csv(f'{DATA_PATH}/copy_list.csv', index=False)
    print(df)

# copy_images()


def count_copied_images():
    copy_list = pd.read_csv(PATH_COPY_LIST).values

    for p_folder, h3, h4 in copy_list:
        p_folder = str(p_folder)
        if h3 == 1 and h4 == 1:
            patient_folder_path_hunt3 = get_local_patient_data_folder(3, p_folder[8:], True)
            patient_folder_path_hunt4 = get_local_patient_data_folder(4, p_folder[8:], True)
            
            if (fc.verify_file(os.path.join(patient_folder_path_hunt4, INAME.format(p_folder[8:], 1, "RAW"))) and
                fc.verify_file(os.path.join(patient_folder_path_hunt3, INAME.format(p_folder[8:], 0, "RAW")))):
                    continue
            elif (fc.verify_file(os.path.join(patient_folder_path_hunt4, INAME.format(p_folder[8:], 1, "BFC"))) and
                 fc.verify_file(os.path.join(patient_folder_path_hunt3, INAME.format(p_folder[8:], 0, "BFC")))):
                    continue
            else:
                print(f"{p_folder} not copied")
                

# count_copied_images()


# for root, dirs, files in os.walk(f"{base_path}/images/data_10"):
#     print(root)
#     print(dirs)
#     print(files)
#     print("")    

# # PATH_HUNT_3_AND_4_INFO = f"{current_path}/original_data/GeneralInfo_HUNT_MRI_3_4.xlsx"
# # hunt_3_4_df = pd.read_excel(PATH_HUNT_3_AND_4_INFO)
# data_folder = "data_missing"
# data_path = f"{base_path}/images/{data_folder}"
# data_path_sort = f"/images/{data_folder}"

# extension_nifti = "nii"
# initial_patient_available = "9410000000039"
# last_patient_available = "9410000008332"

# PATH_HARD_DISK_HUNT4 = f"/media/agustin/Expansion/HUNT-MRI_3&4"
# PATH_HARD_DISK_HUNT3 = f"/media/agustin/LaCie"
# # names_dict_hunt4 = {"T1": "40002_ORIG_Sag_3D_MPRAGE", "T1_BFC": "2_Sag_3D_MPRAGE"}
# # names_dict_hunt3 = {"T1": "3", "T1_BFC": "9"}
# names_dict_hunt4 = {"T1_BFC": "2_Sag_3D_MPRAGE"}
# names_dict_hunt3 = {"T1_BFC": "9"}


# def get_HUNT_data_folder(hunt, absolute=True):
#     if absolute:
#         return f"{data_path}/HUNT{hunt}"
#     else:
#         return f"{data_path_sort}/HUNT{hunt}"


# ## ---------------------- COPY AND CONVERT TO NIFTI PARALEL HUNT

# def get_hard_disk_patient_folders(hunt, path_hard_disk):
#     # read hard disk elements
#     possible_folders = os.listdir(path_hard_disk)

#     # obtain hard disk folders
#     patient_folders = [e for e in possible_folders if os.path.isdir(os.path.join(path_hard_disk, e))]

#     # obtain only folders starting with 9410 (patient folders)
#     if hunt == 4:
#         patient_folders = [e.split("_")[0] for e in patient_folders if e.startswith("9410")]
#     elif hunt == 3:
#         patient_folders = [e for e in patient_folders if e.startswith("9410")]

#     return patient_folders


# def copy_parallel_hunt_and_convert_to_nifti():
#     """
#     Copy and convert to nifti images from both hard disks in parallel.
#     a copy list is returned with the following format:
#     ID, HUNT3, HUNT4
#     ID: patient id
#     HUNT3: 1 if copied, 0 if not
#     HUNT4: 1 if copied, 0 if not
#     """
#     copy_list = np.zeros((0, 3), dtype=int)

#     patient_folders_HUNT3 = get_hard_disk_patient_folders(3, PATH_HARD_DISK_HUNT3)
#     patient_folders_HUNT4 = get_hard_disk_patient_folders(4, PATH_HARD_DISK_HUNT4)

#     # obtain only patients that are in both hard disks
#     patient_folders = list(set(patient_folders_HUNT3) & set(patient_folders_HUNT4))

#     # select available patients (due to some empty folders)
#     patient_folders.sort()
#     initial_patient_index = patient_folders.index(initial_patient_available)
#     last_patient_index = patient_folders.index(last_patient_available)
#     patient_folders = patient_folders[initial_patient_index:last_patient_index+1]

#     # copy and convert to nifti
#     for _patient_folder in tqdm.tqdm(patient_folders):
#         copy_list = np.vstack((copy_list, np.array([int(_patient_folder), -1, -1])))
#         # Rest of the code
#         for hunt in [3, 4]:
#             patient_id = _patient_folder[8:]
#             if hunt == 4:
#                 path_hard_disk = PATH_HARD_DISK_HUNT4
#                 middle_path = None
#                 names_dict = names_dict_hunt4
#                 patient_folder = _patient_folder + "_1"
#             else:
#                 path_hard_disk = PATH_HARD_DISK_HUNT3
#                 middle_path = "1"
#                 names_dict = names_dict_hunt3
#                 patient_folder = _patient_folder

#             original_patient_folder_path = os.path.join(path_hard_disk, patient_folder) 
#             patient_folder_path = os.path.join(get_HUNT_data_folder(hunt, True), patient_id) 

#             # verify patient folder (create if do not exists)
#             if fc.verify_folder(os.path.join(get_HUNT_data_folder(hunt, False), patient_id), create=True):
#                 for m_name, m_folder in names_dict.items():
#                     if middle_path is not None:
#                         original_image_folder = os.path.join(original_patient_folder_path, middle_path, m_folder)
#                     else:
#                         original_image_folder = os.path.join(original_patient_folder_path, m_folder) 
#                     dicom2nifti.convert_directory(original_image_folder, patient_folder_path, compression=False, reorient=True)
#                     # change image name
#                     nifti_image_old_name = os.path.join(patient_folder_path, f'{m_folder.lower()}.{extension_nifti}') 
#                     nifti_image_new_name = os.path.join(patient_folder_path, f'{patient_id}_{hunt-3}_{m_name}.{extension_nifti}') 
#                     try:
#                         os.rename(nifti_image_old_name, nifti_image_new_name)
#                         copy_list[-1, 1 + hunt-3] = 1
#                     except:
#                         print(f"ERROR {_patient_folder} HUNT{hunt}")
#                         copy_list[-1, 1 + hunt-3] = 0
                    
#     return copy_list



# # la mayoria de los erores de HUNT3 son por que no existe la carpeta con el bias field corrected (9) o porque la carpeta existe y no tiene imagenes
# # la mayoria de los errores de HUNT4 son porque la carpeta 2_Sag_3D_MPRAGE comienza por 4 o 5, es decir 4_Sag_3D_MPRAGE o 5_Sag_3D_MPRAGE o cualquier otro numero
# def removed_not_copied_images_folder_and_pairs():
#     """
#     Remove not copied images and folders.
#     the copy list is read and the patiets that do not exists in both HUNT3 and HUNT4 are removed.
#     """
#     copy_list = pd.read_csv(f'{data_path}/copy_list.csv')
#     ids_to_remove = copy_list[(copy_list["HUNT3"] == 0) | (copy_list["HUNT4"] == 0)]["ID"].values
#     folders_to_remove = [str(e)[8:] for e in ids_to_remove if str(e).startswith("9410")]
#     for patient_folder in folders_to_remove:
#         try:
#             shutil.rmtree(os.path.join(get_HUNT_data_folder(3, True), patient_folder))
#             shutil.rmtree(os.path.join(get_HUNT_data_folder(4, True), patient_folder))
#             print(f"Removed {patient_folder}")
#         except:
#             print(f"Already removed {patient_folder}")


# def copy_data_from_hard_disk():
#     """
#     Perform the copy and and the removal of not copied images and folders.
#     """
#     copy_list = copy_parallel_hunt_and_convert_to_nifti()
#     df = pd.DataFrame(copy_list, columns=["ID", "HUNT3", "HUNT4"])
#     df.to_csv(f'{data_path}/copy_list.csv', index=False)

#     removed_not_copied_images_folder_and_pairs()



# ## ---------------------- FIND COPIED IMAGES WITH DIFFERENT DIMENSIONS
# def find_TI_BFC_in_different_folder_HUNT3(patient_id):
#     try:
        
#         original_patient_folder_path = os.path.join(PATH_HARD_DISK_HUNT3, f"94100000{patient_id}","1") 
#         possible_folders = os.listdir(original_patient_folder_path)
#         possible_folders = [int(e) for e in possible_folders if os.path.isdir(os.path.join(original_patient_folder_path, e))]
#         possible_folders.sort()

#         initial_folder = possible_folders.index(9)
#         initial_folder = possible_folders.index(9)
#         possible_folders = possible_folders[initial_folder:]

#         for folder in possible_folders:
#             dicom_folder = os.path.join(original_patient_folder_path, str(folder))
#             dicom_files = [file for file in os.listdir(dicom_folder) if file.endswith('.dcm')]
#             count = len(dicom_files)
#             if 160 < count < 170:
#                 return folder
#     except:
#         pass
#     return None

# def find_images_with_different_dimensions():
#     copy_list = pd.read_csv(f'{data_path}/copy_list_new.csv')
#     np_list = copy_list.values
#     hunt = 3
#     c=0

#     for i, row_list in enumerate(np_list):
#         if row_list[1] == 1 and row_list[2] == 1:
#             patient_folder = str(row_list[0])
#             patient_id = patient_folder[8:]
         
#             img, affine = nfc.load_nifti(f"{get_HUNT_data_folder(hunt, True)}/{patient_id}/{patient_id}_{hunt-3}_T1_BFC.nii")
#             img_size = img.shape
#             if img_size[-1] < 160:
#                 np_list[i, 1] = 0
#                 c+=1

#                 # different_folder = find_TI_BFC_in_different_folder_HUNT3(patient_id)
#                 # if different_folder is None:
#                 #     print(f"{patient_id} HUNT{hunt} {img_size}")
#                 # else:
#                 #     print(f"{patient_id} HUNT{hunt} {img_size} -> {different_folder}")

#                 # 11,
#     df = pd.DataFrame(copy_list, columns=["ID", "HUNT3", "HUNT4"])
#     df.to_csv(f'{data_path}/copy_list_new2.csv', index=False)
      
#     print(c)
# # find_images_with_different_dimensions()
    
# # 33
# ## ---------------------- FIND IMAGES WITH DIFFERENT DIMENSIONS IN HARD DISK

# def copy_and_convert_to_nifty_incomplete_data(_patient_folder, new_dicom_folder):

#     # Rest of the code
#     for hunt in [3, 4]:
#         patient_id = _patient_folder[8:]
#         if hunt == 4:
#             path_hard_disk = PATH_HARD_DISK_HUNT4
#             middle_path = None
#             names_dict = names_dict_hunt4
#             patient_folder = _patient_folder + "_1"
#         else:
#             path_hard_disk = PATH_HARD_DISK_HUNT3
#             middle_path = "1"
#             # names_dict = names_dict_hunt3
#             names_dict = {"T1_BFC": str(new_dicom_folder)}

#             patient_folder = _patient_folder

#         original_patient_folder_path = os.path.join(path_hard_disk, patient_folder) 
#         patient_folder_path = os.path.join(get_HUNT_data_folder(hunt, True), patient_id) 

#         # verify patient folder (create if do not exists)
#         fc.create_folders(os.path.join(get_HUNT_data_folder(hunt, False), patient_id))
#         if fc.verify_folder(os.path.join(get_HUNT_data_folder(hunt, True), patient_id), create=False):
#             for m_name, m_folder in names_dict.items():
#                 if middle_path is not None:
#                     original_image_folder = os.path.join(original_patient_folder_path, middle_path, m_folder)
#                 else:
#                     original_image_folder = os.path.join(original_patient_folder_path, m_folder) 
#                 dicom2nifti.convert_directory(original_image_folder, patient_folder_path, compression=False, reorient=True)
#                 # change image name
#                 nifti_image_old_name = os.path.join(patient_folder_path, f'{m_folder.lower()}.{extension_nifti}') 
#                 nifti_image_new_name = os.path.join(patient_folder_path, f'{patient_id}_{hunt-3}_{m_name}.{extension_nifti}') 
#                 try:
#                     os.rename(nifti_image_old_name, nifti_image_new_name)
#                 except:
#                     print(f"ERROR {_patient_folder} HUNT{hunt}")
#                     return False              
#     return True


# def find_images_with_different_dimensions_hard_disk():
#     copy_list = pd.read_csv(f'{data_path}/copy_list.csv')
#     np_list = copy_list.values
#     c=0
    
#     for i, row_list in enumerate(copy_list.values):
#         if row_list[1] == 0 and row_list[2] == 1:
#             patient_folder = str(row_list[0])
#             different_folder = find_TI_BFC_in_different_folder_HUNT3(patient_folder[8:])
#             if different_folder is None:
#                 print(f"{patient_folder[8:]} HUNT{3}")
#                 c+=1
#             else:
#                 copied = copy_and_convert_to_nifty_incomplete_data(patient_folder, different_folder)
#                 if copied:
#                     np_list[i, 1] = 1
#                 print(f"{patient_folder[8:]} HUNT{3} -> {different_folder} = {np_list[i, 1]}")

#     df = pd.DataFrame(copy_list, columns=["ID", "HUNT3", "HUNT4"])
#     df.to_csv(f'{data_path}/copy_list_new.csv', index=False)
  

#     print(c)

# # find_images_with_different_dimensions_hard_disk()
