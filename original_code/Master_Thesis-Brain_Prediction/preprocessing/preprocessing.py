
import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../'))
sys.path.append(base_path)

from tqdm import tqdm

import project_images_loader as pil

import prep_synthSR as synthSR
import prep_skull_stripping as skull_stripping
import prep_normalization as normalization
import prep_registration as registration
import prep_segment as segment
import prep_segment_merge_tissues as prep_segment_merge_tissues
import prep_mask_no_cerebellum as prep_mask_no_cerebellum



# pil.FOLDER_DATA = pil.FOLDER_DATA_393

matching_ids = pil.get_matching_ids()
progress_bar = tqdm(total=len(matching_ids), desc=f'Initializing...')
for id in matching_ids:
    progress_bar.update(1)
    for hunt in pil.HUNTS:
        
        # print(f"HUNT{hunt} {id}")

        # # # ----------------  Step 1: SynthSR ----------------
        # progress_bar.set_description(desc=f"Step 1: SynthSR HUNT{hunt} {id}")
        # synthSR.save_synthSR(hunt, id, verify=True)
     
        # # # ----------------  Step 2: Registration ----------------
        # progress_bar.set_description(desc=f"Step 2.1: Registration to MNI HUNT{hunt} {id}")
        # registration.save_registration_to_MNI_freesurfer(hunt, id)
        # # if hunt == 4:
        # #     progress_bar.set_description(desc=f"Step 2.2: Registration to HUNT3 HUNT{hunt} {id}")
        # #     registration.save_registration_to_hunt3_freesurfer(id)

        # # # ---------------- Step 3: Skull stripping ----------------
        # progress_bar.set_description(desc=f"Step 3: Skull stripping HUNT{hunt} {id}")
        # skull_stripping.save_skull_stripping_freesurfer(hunt, id, pil.REGISTRATION_TO_MNI)

        # # # ---------------- Step 4: Normalization ----------------
        # progress_bar.set_description(desc=f"Step 4: Normalization HUNT{hunt} {id}")
        # normalization.save_normalized_image(hunt, id) #RECUERDA ELIMINAR LOS NOGATIVOS USANDO LA VENTANA MEAN

        # # # ---------------- Step 5: Segmentation ----------------
        # # uncomment the following in the FASTsurfer computer

        # # # Step 5.1: FASTSURFER Segmentation
        # progress_bar.set_description(desc=f"Step 5.1: Segmentation HUNT{hunt} {id}")
        # segmentation_folder_mgz = f"C:/Users/Agustin/Master/Master/S4/code/images/segmentations/segmentation_mgz"
        # segment.segment_freesurfer(hunt, id, GPU=True, segmentation_folder=segmentation_folder_mgz, verify=True)

        # # # Step 5.2: Registration to patient space
        # # copy the segmentation_data_..._mgz in the FREEsurfer computer and uncomment the followin lines
        # progress_bar.set_description(desc=f"Step 5.2: Segmentation patient reg HUNT{hunt} {id}")
        # segmentation_folder_mgz = f"/home/agustin/Documents/S4/code/images/segmentations/segmentation_data_missing_mgz"
        # segmentation_folder_nii = f"/home/agustin/Documents/S4/code/images/segmentations/segmentation_data_missing_nii"
        # segment.transform_segmentation_to_patient_space(hunt, id, segmentation_folder_mgz, segmentation_folder_nii)

        # # # Step 5.3: copy in patient folder
        # # # copy the segmentation_data_..._nii in the FASTsurfer computer and uncomment the followin lines
        # progress_bar.set_description(desc=f"Step 5.3: Segmentation Copy HUNT{hunt} {id}")
        # # segmentation_folder_nii = f"/home/agustin/Documents/S4/code/images/segmentations/segmentation_data_missing_nii"
        # segmentation_folder_nii = f"C:/Users/Agustin/Master/Master/S4/code/images/segmentations/segmentation_data_missing_nii"
        # segment.copy_seg_to_patient_folder(hunt, id, segmentation_folder_nii)

        # # # Step 6: Create 3 tissues segmentation
        # progress_bar.set_description(desc=f"Step 6: Create 3 tissues seg HUNT{hunt} {id}")
        # prep_segment_merge_tissues.save_3_tissues_segmentation(hunt, id, verify=False)

        # # # Step 7: Create brain mask without cerebellum
        progress_bar.set_description(desc=f"Step 7: Create brain mask without cerebellum HUNT{hunt} {id}")
        prep_mask_no_cerebellum.create_mask_no_cerebellum(hunt, id)
        break
    break
