import glob
import os
import shutil



def copy_segmentation_to_convert():
                
    source_dir = "C:/Users/Agustin/Master/Master/S4/code/images/data_393/"
    target_dir = "C:/Users/Agustin/Master/Master/S4/code/images/segmentations/segmentation_data_393_mgz/"

    # Copy images
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if (file.endswith("SEG_SIMPLE.mgz") or
                file.endswith("SEG.mgz")):
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_dir, os.path.relpath(root, source_dir), file)
                print(f"Copying {source_file} to {target_file}")
                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                shutil.copy2(source_file, target_file)


def copy_converted_segmentation():
                    
    source_dir = "/home/agustin/Documents/S4/code/images/data_393/"
    target_dir = "/home/agustin/Documents/S4/code/images/segmentations/segmentation_data_393_nii/"

    # Copy images
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if (file.endswith("SEG_SIMPLE.nii.gz") or
                file.endswith("SEG.nii.gz")):
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_dir, os.path.relpath(root, source_dir), file)
                print(f"Copying {source_file} to {target_file}")
                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                shutil.copy2(source_file, target_file)



def remove_mgz_files():
    source_dir = "/home/agustin/Documents/S4/code/images/data_393/"
    # source_dir = "C:/Users/Agustin/Master/Master/S4/code/images/data_393/"

    # Copy images
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if (file.endswith("SEG_SIMPLE.mgz") or
                file.endswith("SEG.mgz")):
                source_file = os.path.join(root, file)
                os.remove(source_file)
          
# remove_mgz_files()