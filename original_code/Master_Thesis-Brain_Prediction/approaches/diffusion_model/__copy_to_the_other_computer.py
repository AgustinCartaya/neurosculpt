import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(base_path)

import glob

import shutil




source_dir = f"{current_path}/images/results_postprocessed"
target_dir = f"{current_path}/images/results_postprocessed_2copy"


# Copy images
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if (file.endswith("0_SEG_96_POST_RES.mgz")):
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_dir, os.path.relpath(root, source_dir), file)
            print(f"Copying {source_file} to {target_file}")
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            shutil.copy2(source_file, target_file)
