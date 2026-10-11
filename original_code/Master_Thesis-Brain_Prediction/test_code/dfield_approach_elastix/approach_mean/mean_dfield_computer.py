import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../../'))
dfield_approach_elastix_path = os.path.abspath(os.path.join(current_path, '../'))

sys.path.append(base_path)
sys.path.append(dfield_approach_elastix_path)

from tqdm import tqdm
import numpy as np
import project_images_loader as pil
import utils.nifti_functions as nfc

import constants as cts

def compute_mean_dfield(pfile_number, nb_images, exclude_ids=None):
    patient_ids = pil.get_matching_ids()
    dfield_sum = None
    for i in tqdm(range(nb_images)):
        id = patient_ids[i]
        if exclude_ids is not None and id in exclude_ids:
            print(f"Excluding {id} from mean dfield computation")
            continue
        _dfield, affine = nfc.load_nifti(cts.get_image_path_name(id, pfile_number, cts.INAME_DFIELD, subfolder=cts.FOLDER_REGISTRATIONS))
        if dfield_sum is None:
            dfield_sum = _dfield
        else:
            dfield_sum += _dfield
                
    dfield_sum /= nb_images
    return dfield_sum.astype(np.float32), affine

def save_mean_dfield(pfile_number, nb_images, exclude_ids=None):
    dfield_mean, affine = compute_mean_dfield(pfile_number, nb_images, exclude_ids=exclude_ids)
    nfc.save_nifti(dfield_mean, affine, cts.get_mean_dfield_path_name(pfile_number, nb_images))


save_mean_dfield(cts.PFILE_0038, 10, pil.EXCLUDED_PATIENTS + ["08202"])
save_mean_dfield(cts.PFILE_0038, 100, pil.EXCLUDED_PATIENTS + ["08202"])
save_mean_dfield(cts.PFILE_0038, 200, pil.EXCLUDED_PATIENTS + ["08202"])
save_mean_dfield(cts.PFILE_0038, 300, pil.EXCLUDED_PATIENTS + ["08202"])
save_mean_dfield(cts.PFILE_0038, 620, pil.EXCLUDED_PATIENTS + ["08202"])
    
