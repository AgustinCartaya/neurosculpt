import glob
import os
import shutil




source_dir = "/home/agustin/Documents/S4/code/images/data_complete_old"
target_dir = "/home/agustin/Documents/S4/code/images/data_complete"

# Copy images
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if (file.endswith("T1.nii") or
            file.endswith("T1_SYNTHSR.nii")):
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_dir, os.path.relpath(root, source_dir), file)
            print(f"Copying {source_file} to {target_file}")
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            shutil.copy2(source_file, target_file)

