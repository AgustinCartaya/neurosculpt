
import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(base_path)


import subprocess
import project_images_loader as pil

def save_deformation_field_voxelmorph(id):
    # ./scripts/tf/register.py --moving moving.nii.gz --fixed atlas.nii.gz --moved warped.nii.gz --model model.h5 --gpu 0
    # ./scripts/tf/register.py --moving moving.nii.gz --fixed atlas.nii.gz --moved warped.nii.gz --model model.h5 --gpu 0 --save-warp
    
    # model_path_name = f"{current_path}/voxelmorph-dev/models/vxm_dense_brain_T1_3D_mse.h5"
    # model_path_name = f"{current_path}/voxelmorph-dev/models/brains-dice-vel-0.5-res-16-256f.h5"
    model_path_name = f"{current_path}/voxelmorph-dev/models/synthmorph.deform.2.h5"
    image_res_path_name = f"{current_path}/images/{id}_0_T1_VOXELMORPH3.nii.gz"
    wrap_res_path_name = f"{current_path}/images/{id}_0_DFIELD_VOXELMORPH3.nii.gz"
    command = ["python", f"{current_path}/voxelmorph-dev/scripts/tf/register.py", 
               "--moving", pil.get_preprocessed_image_path_name(3, id, absolute=True), 
               "--fixed", pil.get_preprocessed_image_path_name(4, id, absolute=True),
               "--moved", image_res_path_name,
               "--model", model_path_name,
            #    "--gpu", "0",
                "--warp", wrap_res_path_name,
                "--image_size", "192,224,192"
                ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        # Print the output
        print(f"Deformation field obtained {id}")
        # print(result.stdout)
    else:
        # Print error message
        print("Error:", result.stderr)

save_deformation_field_voxelmorph("00039")
        





