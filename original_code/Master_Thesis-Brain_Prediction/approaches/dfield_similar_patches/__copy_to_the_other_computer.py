import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(base_path)

import glob

import shutil




source_dir = f"{current_path}/images/results_postprocessed"
target_dir = f"{current_path}/images/results_postprocessed"


# Copy images
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if (file.endswith("0_BMASK_NCRB_96_10.nii.gz")):
            parts = file.split("_")
            parts.insert(4, "POST")
            new_name = "_".join(parts)
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_dir, os.path.relpath(root, source_dir), new_name)
            print(f"Copying {source_file} to {target_file}")
            os.rename(source_file, target_file)
