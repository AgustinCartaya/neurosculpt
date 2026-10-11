
import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../'))
sys.path.append(base_path)

import numpy as np
import utils.functions as fc
import utils.util_freesurfer_segmentation as ufs
import utils.nifti_functions as nfc
import project_images_loader as pil


def save_3_tissues_segmentation(hunt, id, verify=True):
    if verify:
        if fc.verify_file(pil.get_segmentation_path_name(hunt, id, nb_tissues=3, absolute=True)):
            print(f"3 tissues already done for HUNT{hunt} {id}")
            return True
        
    seg, aff = nfc.load_nifti(pil.get_segmentation_path_name(hunt, id, nb_tissues=36, absolute=True))
    seg_3 = ufs.merge_seg36_to_seg3(seg)
    nfc.save_nifti(seg_3, aff, pil.get_segmentation_path_name(hunt, id, nb_tissues=3, absolute=True))
