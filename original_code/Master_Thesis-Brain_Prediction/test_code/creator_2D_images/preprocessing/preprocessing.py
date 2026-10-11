
import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(base_path)

from tqdm import tqdm

import project_images_loader as pil


import prep_normalization as normalization
import prep_registration as registration



# pil.FOLDER_DATA = pil.FOLDER_DATA_393

matching_ids = pil.get_matching_ids()
progress_bar = tqdm(total=len(matching_ids), desc=f'Initializing...')
for id in matching_ids:
    progress_bar.update(1)
    for hunt in pil.HUNTS:
        
        # print(f"HUNT{hunt} {id}")

        # # # # ----------------  Step 1: Registration ----------------
        # progress_bar.set_description(desc=f"Step 1: Registration to MNI HUNT{hunt} {id}")
        # registration.save_registration_to_MNI_elastix(hunt, id)

        # # # ----------------  Step 2: Tranformation ----------------
        # progress_bar.set_description(desc=f"Step 2: Tranformation to MNI HUNT{hunt} {id}")
        # registration.save_transformation_mask_to_MNI(hunt, id)

        # # # ---------------- Step 3: Normalization ----------------
        progress_bar.set_description(desc=f"Step 3: Normalization HUNT{hunt} {id}")
        normalization.save_normalized_image(hunt, id) #RECUERDA ELIMINAR LOS NOGATIVOS USANDO LA VENTANA MEAN
        

