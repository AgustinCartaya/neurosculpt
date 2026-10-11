import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))

sys.path.append(base_path)


import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
from tqdm import tqdm
import pandas as pd
from IPython.display import display


import project_images_loader as pil
import utils.util as util
import utils.functions as fc
import utils.nifti_functions as nfc

import constants as cts
import utils.dfield_applier as dfield_applier



def register_val_images(img_path_name, dfield_path_name, binary=False):
    img_reg = dfield_applier.apply_deformation_field_elastix(img_path_name, dfield_path_name, binary=binary)
    if binary:
        img_reg = img_reg.astype(np.uint16)
    return img_reg

def save_result(img_reg, result_path_name, original_path_name):
    if fc.verify_folder(os.path.dirname(result_path_name), create=True):
        affine = nfc.load_nifti(original_path_name)[1]
        nfc.save_nifti(img_reg, affine, result_path_name)

def register_all_val_images(args):
    val_images = ["08202"] + pil.get_matching_ids()[621:]
    for id in tqdm(val_images):
        # original image path
        img_path_name = pil.get_patient_image_type_path_name(3, id, args["img_load_type"], args["img_load_ext"], absolute=True)
        # dfield path
        dfield_path_name = cts.get_dfield_path_name(args["nb_images"])
        # result image path
        img_reg_path_name = cts.get_result_image_path_name(id, args["img_save_type"], (args["nb_images"],), absolute=True)

        # register image and save
        img_reg = register_val_images(img_path_name, dfield_path_name, binary=args["binary"])
        save_result(img_reg, img_reg_path_name, img_path_name)

def create_all():
    for nb_images in [10, 100, 200, 300, 620]:
        print(f"-------- Registering images using MEAN DFIELD whit {nb_images} images")
        # print("T1")
        # args = {"img_load_type":"T1_MNI", "img_load_ext":"nii", "img_save_type":"T1", "nb_images":nb_images, "binary":False}
        # register_all_val_images(args)
        # print("SEG_96")
        # args = {"img_load_type":"SEG_96", "img_load_ext":"nii.gz", "img_save_type":"SEG_96", "nb_images":nb_images, "binary":True}
        # register_all_val_images(args)
        print("BMASK_NCRB")
        args = {"img_load_type":"BMASK_NCRB_MNI", "img_load_ext":"nii.gz", "img_save_type":"BMASK_NCRB", "nb_images":nb_images, "binary":True}
        register_all_val_images(args)

create_all()