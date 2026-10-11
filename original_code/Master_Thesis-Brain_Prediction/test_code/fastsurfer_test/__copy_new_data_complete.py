
import sys
import os
import shutil

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(base_path)

import numpy as np
import subprocess
import project_images_loader as pil
import utils.nifti_functions as nfc
import utils.functions as fc
import utils.util as util


source_dir = f"{current_path}/images/output_preprocessed"
target_dir = f"{current_path}/images/output_renamed"

files = [("aparc.DKTatlas+aseg.deep.mgz", "{0}_{1}_SEG.mgz"),
        ("aseg.auto_noCCseg.mgz", "{0}_{1}_SEG_SIMPLE.mgz"),
        # ("cerebellum.CerebNet.nii.gz", "{0}_{1}_SEG_CEREBELLUM.nii.gz"),
        ("mask.mgz", "{0}_{1}_BMASK.mgz"),
        ("orig.mgz", "{0}_{1}_T1_0038.mgz"),
        # ("orig_nu.mgz", "{0}_{1}_T1_0038_BFIELD_CORRECTED.mgz"),
        ]

for id in pil.test_patients_ids:
    for hunt in [3,4]:
        for file in files:
            source_file = os.path.join(source_dir, f"HUNT{hunt}/{id}/mri", file[0])
            target_file = os.path.join(target_dir, file[1].format(id, hunt-3))

            print(f"Copying {source_file} to {target_file}")
            # os.makedirs(os.path.dirname(target_file), exist_ok=True)
            shutil.copy2(source_file, target_file)

