"""Create the PREP_MNI 3D images and the 2D PNG slices used by StarGAN v2."""
import argparse

import numpy as np
from tqdm import tqdm

from brainpred import splits
from brainpred.preprocessing import slices_2d

STEPS_HELP = """\
steps (comma separated, executed in this order for each patient, HUNT3 then HUNT4):
  register   affine elastix (PAR_0038_AFF) of T1_PREP to the preprocessed MNI template -> T1_PREP_MNI + parameter file
  transform  apply it to BMASK_MNI and SEG_3                                          -> BMASK_PREP_MNI, SEG_3_PREP_MNI
  normalize  background to 0, scale T1_PREP_MNI to [0, 1] (in place)
  inverse    inverse parameter file (HUNT3 only), used to bring 2D-method predictions back
  slices     PNG slices LAYER_START_2D..LAYER_END_2D-1 of T1, BMASK, SEG_3
analysis (no files written):
  layers     print the brain start/end slice statistics used to choose LAYER_START_2D / LAYER_END_2D
"""
STEPS = ["register", "transform", "normalize", "inverse", "slices", "layers"]


def get_ids(args):
    if args.ids:
        return args.ids
    return {"train": splits.get_train_ids, "validation": splits.get_validation_ids, "all": splits.get_matching_ids}[args.split]()


def print_layer_statistics(ids, min_tissue=0.15):
    layer_start_end = slices_2d.create_start_end_list(ids, min_tissue=min_tissue)
    for name, fn in [("min", np.min), ("max", np.max), ("mean", np.mean), ("median", np.median), ("std", np.std)]:
        print(f"{name}_slide: {fn(layer_start_end, axis=0)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=STEPS_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", required=True, help=f"comma separated subset of: {','.join(STEPS)}")
    parser.add_argument("--ids", nargs="+", help="patient IDs (default: the whole --split)")
    parser.add_argument("--split", choices=["train", "validation", "all"], default="all")
    parser.add_argument("--axis", type=int, choices=[0, 2], default=0,
                        help="slicing axis: 0 = sagittal folder (default, used by StarGAN v2), 2 = axial folder (last call in the original notebook)")
    args = parser.parse_args()

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = set(steps) - set(STEPS)
    if unknown:
        parser.error(f"unknown steps: {sorted(unknown)}")

    ids = get_ids(args)
    for id in tqdm(ids):
        for hunt in splits.HUNTS:
            if "register" in steps:
                slices_2d.save_registration_to_MNI_elastix(hunt, id)
            if "transform" in steps:
                slices_2d.save_transformation_mask_to_MNI(hunt, id)
            if "normalize" in steps:
                slices_2d.save_normalized_image(hunt, id)
            if "inverse" in steps and hunt == 3:
                slices_2d.find_inverse_registration_param_files(hunt, id)
            if "slices" in steps:
                slices_2d.create_and_save_2D_images_patient(hunt, id, axis=args.axis)

    if "layers" in steps:
        print_layer_statistics(ids)


if __name__ == "__main__":
    main()
