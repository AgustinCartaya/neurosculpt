
import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../'))
sys.path.append(base_path)

import subprocess
import project_images_loader as pil
import utils.functions as fc

def save_synthSR(hunt, id, verify=False):
    if verify:
        exist = fc.verify_file(pil.get_synthSR_image_path_name(hunt, id, absolute=True))
        if exist:
            print(f"SynthSR already done for HUNT{hunt} {id}")
            return True
 
    command = ["mri_synthsr", 
               "--i", pil.get_original_image_path_name(hunt, id, absolute=True), 
               "--o", pil.get_synthSR_image_path_name(hunt, id, absolute=True), 
               "--threads", "1"]
    
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        # Print the output
        print(f"Freesurfer synthSR done for HUNT{hunt} {id}")
        # print(result.stdout)
    else:
        # Print error message
        print("Error:", result.stderr)


