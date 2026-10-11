

import subprocess
# import project_images_loader as pil
# import utils.functions as fc

def save_synthSR(path_name_in, path_name_out, verify=False):
    command = ["mri_synthsr", 
               "--i", path_name_in, 
               "--o", path_name_out, 
               "--threads", "1"]
    
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        # Print the output
        print(f"Freesurfer synthSR done for \n {path_name_in}")
        # print(result.stdout)
    else:
        # Print error message
        print("Error:", result.stderr)


