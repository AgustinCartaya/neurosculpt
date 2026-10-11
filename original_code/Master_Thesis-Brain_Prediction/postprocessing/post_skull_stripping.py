
import os

import subprocess
# import project_images_loader as pil



def save_skull_stripping_freesurfer(path_name_in, mask_out, path_name_out=None):

    if path_name_out is None:
        _path_name_out = path_name_in + "rm.nii.gz"

    command = ["mri_synthstrip", 
               "-i", path_name_in, 
               "-o", _path_name_out,
               "-m", mask_out ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        # Print the output
        if path_name_out is None:
            os.remove(_path_name_out)
        print(f"Freesurfer skull stripping done for {path_name_in}")
    else:
        print("Error:", result.stderr)

