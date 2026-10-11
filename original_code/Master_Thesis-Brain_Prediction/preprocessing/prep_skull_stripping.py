
import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../'))
sys.path.append(base_path)

import subprocess
import project_images_loader as pil



def save_skull_stripping_freesurfer(hunt, id, registration_to):
    # mri_synthstrip -i 12082_0_T1_SYNTHSR.nii -o stripped2.nii -m mask.mgz --no-csf
    if registration_to == pil.REGISTRATION_TO_HUNT3 and hunt == pil.HUNTS[0]:
        # command = ["mri_synthstrip", "-i", pil.get_synthSR_image_path_name(hunt, id), "-o", pil.get_patient_image_path_name(hunt, id, "nsk.nii", absolute=True), "--no-csf"]
        command = ["mri_synthstrip", "-i", pil.get_synthSR_image_path_name(hunt, id), "-o", pil.get_patient_image_path_name(hunt, id, "nsk.nii", absolute=True)]
    else:
        # command = ["mri_synthstrip", "-i", pil.get_registered_to_template_image_path_name(hunt, id, registration_to), "-o", pil.get_patient_image_path_name(hunt, id, "nsk.nii", absolute=True), "--no-csf"]
        command = ["mri_synthstrip", "-i", pil.get_registered_to_template_image_path_name(hunt, id, registration_to), "-o", pil.get_patient_image_path_name(hunt, id, "nsk.nii", absolute=True)]

    command.extend(["-m", pil.get_brain_mask_path_name(hunt, id, registration_to)])
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        # Print the output
        os.remove(pil.get_patient_image_path_name(hunt, id, "nsk.nii", absolute=True))
        print(f"Freesurfer skull stripping done for HUNT{hunt} {id}")

        # print(result.stdout)
    else:
        # Print error message
        print("Error:", result.stderr)

