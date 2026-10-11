import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../../../'))

sys.path.append(base_path)

import numpy as np
import pandas as pd
from tqdm import tqdm

import project_images_loader as pil

import utils.util as util
import utils.nifti_functions as nfc
import test_code.dfield_approach_elastix.constants as cts
import utils.util_freesurfer_segmentation as ufs
import utils.functions as fc


INAME_RESULT_IMAGE = "{0}_0_{1}_{2}.nii.gz"


def load_original_segmentation(id, hunt, nb_tissues):
    seg, _ = nfc.load_nifti(pil.get_segmentation_path_name(hunt, id, nb_tissues=96, absolute=True))
    if nb_tissues == 3:
        seg = ufs.remove_cerebellum(ufs.merge_seg96_to_seg3(seg), seg, deep_copy=False)
    elif nb_tissues == 36:
        seg = ufs.merge_seg96_to_seg36(seg)
    return seg


def load_result_segmentation(id, nb_tissues, approach, extra_info):
    extra_info_str = "_".join([str(info) for info in extra_info])
    path = f"{base_path}/approaches/{approach}/images/results_postprocessed/{id}"
    img_sufix = "_RAW" if approach == "MPGAN_paper_16_seg_loss" else "_POST"

    seg, _ = nfc.load_nifti(f"{path}/{INAME_RESULT_IMAGE.format(id, 'SEG_96'+img_sufix, extra_info_str)}")
    if nb_tissues == 3:
        seg = ufs.remove_cerebellum(ufs.merge_seg96_to_seg3(seg), seg, deep_copy=False)
    elif nb_tissues == 36:
        seg = ufs.merge_seg96_to_seg36(seg)
    return seg




def compute_patient_metrics(id, nb_tissues, approach, extra_info):
    seg_hunt3 = load_original_segmentation(id, 3, nb_tissues=nb_tissues)
    seg_hunt4 = load_original_segmentation(id, 4, nb_tissues=nb_tissues)
    seg_result = load_result_segmentation(id, nb_tissues, approach, extra_info)

    seg_indexs = np.unique(seg_hunt4.astype(np.uint))
    seg_indexs = np.sort(seg_indexs)

    tissue_metrics = {}
    for tissue_id in np.unique(seg_indexs):
        mask_hunt3 = seg_hunt3 == tissue_id
        mask_hunt4 = seg_hunt4 == tissue_id
        mask_result = seg_result == tissue_id

        _dice_hunt3 = util.compute_dice_coefficient(mask_result, mask_hunt3)
        _dice_hunt4 = util.compute_dice_coefficient(mask_result, mask_hunt4)
        _vol = np.sum(mask_result)

        tissue_metrics[tissue_id] = (_vol, _dice_hunt3, _dice_hunt4)

    columns_names = ['vol', 'dice_hunt3', 'dice_hunt4'] 
    tissue_metrics_df = pd.DataFrame.from_dict(tissue_metrics, orient='index', columns=columns_names)
    tissue_metrics_df.index.name = 'tissue_id'

    # save results
    path_name = f'{current_path}/features/{approach}/tissues_{nb_tissues}/{id}.csv'
    tissue_metrics_df.to_csv(path_name)

def compute_patient_metrics_for_all(approach, extra_info):
    print(f"Computing metrics for approach {approach} and extra_info {extra_info}")
    if approach == "diffusion_model":
        val_images = pil.get_val_ids_2()
    else:
        val_images = ["08202"] + pil.get_matching_ids()[621:]
    for nb_tissues in [3, 36, 96]:
        print(f'Computing metrics for {nb_tissues} tissues')
        for matching_id in tqdm(val_images):
            compute_patient_metrics(matching_id, nb_tissues, approach, extra_info)
   
    
# compute_patient_metrics_for_all("dfield_similar_patches", [96, 10])
# compute_patient_metrics_for_all("MGAN_paper_16", [46])
# compute_patient_metrics_for_all("MPGAN_paper_16_seg_loss", [52])
# compute_patient_metrics_for_all("StarGAN_V2_sagital", ["BEST"])
# compute_patient_metrics_for_all("diffusion_model", ["RES"])
compute_patient_metrics_for_all("MPGAN_paper_16", [28])
