import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))

sys.path.append(base_path)

import numpy as np
from tqdm import tqdm

import project_images_loader as pil
pil.FOLDER_DATA = pil.FOLDER_DATA_393

import utils.util as util
import utils.functions as fc
import utils.nifti_functions as nfc

def verify_tissues():
    matching_ids = pil.get_matching_ids()
    nb_seg = []
    for matching_id in tqdm(matching_ids):
        for hunt in [3, 4]:
            mask, _ = nfc.load_nifti(pil.get_segmentation_path_name(hunt, matching_id, simple=False, absolute=True))
            nb_seg.append(len(np.unique(mask)))
    print(util.get_basic_metrics(np.array(nb_seg)))

def convert_seg_to_csf_gm_wm(seg):
    # chat 
    # csf_indices = [14]
    # gm_indices = [11, 12, 13, 17, 18, 26, 28, 31, 50, 51, 52, 54, 63, 1005, 1007, 1008, 1009, 1011, 1012, 1013, 1014, 1016, 1017, 1018, 1019, 1021, 1023, 1025, 1027, 1034, 1035, 2005, 2006, 2008, 2009, 2011, 2012, 2014, 2015, 2016, 2018, 2019, 2020, 2021, 2022, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2035]
    # wm_indices = [2, 7, 10, 16, 24, 41, 46, 49, 53, 58, 60, 63, 77, 1002, 1003, 1006, 1010, 1015, 1020, 1022, 1024, 1026, 1028, 1029, 1030, 1031, 1034, 1035, 2002, 2003, 2007, 2010, 2013]

    # hand made not finished
    csf_indices = [14, 24, 4, 43, 63, 31]
    gm_indices = [42, 3, 50, 11, 49, 10, 51, 12, 52, 13, 53, 17]
    wm_indices = [2, 7, 10, 16, 24, 41, 46, 49, 53, 58, 60, 63, 77, 1002, 1003, 1006, 1010, 1015, 1020, 1022, 1024, 1026, 1028, 1029, 1030, 1031, 1034, 1035, 2002, 2003, 2007, 2010, 2013]



    seg_csf = np.where(np.isin(seg, csf_indices), 1, 0)
    seg_gm= np.where(np.isin(seg, gm_indices), 2, 0)
    seg_wm = np.where(np.isin(seg, wm_indices), 3, 0)

    seg_csf_gm_wm = seg_csf + seg_gm + seg_wm
    return seg_csf_gm_wm


def convert_all_seg_to_csf_gm_wm():
    matching_ids = pil.get_matching_ids()
    hunt = 3
    for id in tqdm(matching_ids):
        seg, aff = nfc.load_nifti(pil.get_segmentation_path_name(hunt, id, simple=False, absolute=True))
        seg_csf_gm_wm = convert_seg_to_csf_gm_wm(seg).astype(np.uint8)
        nfc.save_nifti(seg_csf_gm_wm, aff, f"{current_path}/images/{id}/{id}_{hunt}_SEG_3T.nii.gz") 
        break

convert_all_seg_to_csf_gm_wm()