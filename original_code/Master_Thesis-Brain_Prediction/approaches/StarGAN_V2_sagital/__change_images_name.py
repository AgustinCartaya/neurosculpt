import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(base_path)

import glob

import shutil




source_dir = f"{current_path}/images_prep_MNI/results"
target_dir = f"{current_path}/images_prep_MNI/results_2"


# Copy images
for root, dirs, files in os.walk(source_dir):
    for file in files:
        id = file.split("_")[0]
        if (not file.endswith(f"{id}.nii.gz")):
            source_file = os.path.join(root, file)
            new_name = f"{id}_0_T1_BEST.nii.gz"
            target_file = os.path.join(target_dir, os.path.relpath(root, source_dir), new_name)
            print(f"Copying {source_file} to {target_file}")
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            shutil.copy2(source_file, target_file)
