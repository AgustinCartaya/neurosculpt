
import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../'))
sys.path.append(base_path)

import subprocess
import project_images_loader as pil
import utils.functions as fc

import post_synthsr as post_synthsr
import post_skull_stripping as post_skull_stripping
import post_hist_matching as post_hist_matching

def do_for_MGAN_PAPER():
    path_in = f"{base_path}/test_code/MGAN/MGAN_paper/images/first_results"
    path_out = f"{base_path}/test_code/MGAN/MGAN_paper/images/first_results/postprocessed"

    org_name = "{0}_CHECK_{1}.nii.gz"

    prep_name = "{0}_CHECK_{1}{2}.nii.gz"
    bmask_name = "{0}_CHECK_{1}_BMASK.nii.gz"

    for id in ["08202"]:
        for check in [17,19, 23, 25, 27, 29, 21, 31]:
        # for check in [21, 31]:

            # ## ---- Step 1: SynthSR
            # print(f"Step 1: SynthSR for {id} {check}")
            # name_in = org_name.format(id, check)
            # name_out = prep_name.format(id, check, "_SYNTHSR")

            # path_name_in = f"{path_in}/{name_in}"
            # path_name_out = f"{path_out}/{name_out}"

            # post_synthsr.save_synthSR(path_name_in, path_name_out)


            # ## ---- Step 2: Skull stripping
            # print(f"Step 2: Skull stripping for {id} {check}")
            # name_in = prep_name.format(id, check, "_SYNTHSR")
            # bmask_out = bmask_name.format(id, check)

            # path_name_in = f"{path_out}/{name_in}"
            # path_bmask_name_out = f"{path_out}/{bmask_out}"

            # post_skull_stripping.save_skull_stripping_freesurfer(path_name_in, path_bmask_name_out)


            ## ---- Step 3: Histogram matching
            print(f"Step 3: Histogram matching for {id} {check}")
            name_in = prep_name.format(id, check, "_SYNTHSR")
            bmask_in = bmask_name.format(id, check)
            name_out = prep_name.format(id, check, "_POST")

            path_name_ref = pil.get_preprocessed_image_path_name(3, id)
            path_name_in = f"{path_out}/{name_in}"
            path_bmask_in = f"{path_out}/{bmask_in}"
            path_name_out = f"{path_out}/{name_out}"

            post_hist_matching.save_histogram_match(path_name_ref, path_name_in, path_name_out, path_bmask_in)


# do_for_MGAN_PAPER()