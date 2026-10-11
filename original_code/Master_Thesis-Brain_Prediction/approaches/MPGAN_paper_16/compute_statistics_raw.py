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

import utils.util as util
import utils.nifti_functions as nfc
import utils.functions as fc
import utils.util_freesurfer_segmentation as ufs
import utils.util_brain_metrics as ubm
import pandas as pd


import project_images_loader as pil


import constants as cts


def save_stats(stats, ids, columns_names, table_name, postprocessed):
    ssim_df = pd.DataFrame(stats, index=ids, columns=columns_names)
    ssim_df.index.name = 'id'
    ssim_df.to_csv(cts.get_statistics_path_name(table_name, postprocessed=postprocessed), index=True, header=True)



def compute_statistics(ids, checkpoints_list):
    postprocessed=False
    columns_names = [f"{checkpoint}" for checkpoint in checkpoints_list]
    ssim_stats = np.zeros((len(ids), len(columns_names)))
    ncc_stats = np.zeros((len(ids), len(columns_names)))
    psnr_stats = np.zeros((len(ids), len(columns_names)))
    for i, id in enumerate(tqdm(ids)):
        img_org, _ = nfc.load_nifti(pil.get_registered_to_template_image_path_name(4, id, pil.REGISTRATION_TO_MNI, absolute=True))
        for j, checkpoint in enumerate(checkpoints_list):
            img_res, _ = nfc.load_nifti(cts.get_result_image_path_name(id, "T1", (checkpoint,), postprocessed))
            # compute ssim
            ssim_res = util.compute_ssim(img_org, img_res)
            ssim_stats[i, j] = ssim_res
        
            # compute ncc
            ncc_res = util.compute_normalized_cross_correlation(img_org, img_res)
            ncc_stats[i, j] = ncc_res

            # compute psnr
            psnr_res = util.compute_psnr(img_org, img_res)
            psnr_stats[i, j] = psnr_res
        break
    save_stats(ssim_stats, ids, columns_names, "ssim" , postprocessed)
    save_stats(ncc_stats, ids, columns_names, "ncc" , postprocessed)
    save_stats(psnr_stats, ids, columns_names, "psnr" , postprocessed)


val_images = ["08202"] + pil.get_matching_ids()[621:]
compute_statistics(val_images, [27, 28, 30, 32])