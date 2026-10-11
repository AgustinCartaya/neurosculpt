"""Register every patient's HUNT3 scan to its HUNT4 scan with elastix (B-spline, PAR_0038) to obtain the deformation fields."""
import argparse

from tqdm import tqdm

from brainpred import splits
from brainpred.registration import elastix

STEPS_HELP = """\
steps:
  register    elastix HUNT3 T1_MNI -> HUNT4 T1_MNI; saves T1, DFIELD and PARAMFILE (_0038)
  transform   apply the saved transform to BMASK_MNI, SEG_96 and SEG_36 of HUNT3 (nearest neighbour)
Outputs go to <work_dir>/longitudinal_registrations/{id}_0_{TYPE}_0038.nii.gz
"""
STEPS = ["register", "transform"]


def get_ids(args):
    if args.ids:
        return args.ids
    return {"train": splits.get_train_ids, "validation": splits.get_validation_ids, "all": splits.get_matching_ids}[args.split]()


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=STEPS_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", default="register,transform", help=f"comma separated subset of: {','.join(STEPS)}")
    parser.add_argument("--ids", nargs="+", help="patient IDs (default: the whole --split)")
    parser.add_argument("--split", choices=["train", "validation", "all"], default="all")
    args = parser.parse_args()

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = set(steps) - set(STEPS)
    if unknown:
        parser.error(f"unknown steps: {sorted(unknown)}")

    progress_bar = tqdm(get_ids(args))
    for id in progress_bar:
        progress_bar.set_description(desc=f"Registering {id} HUNT3 to HUNT4")
        if "register" in steps:
            elastix.register_longitudinal(id, transform=False)
        if "transform" in steps:
            elastix.transform_seg_and_mask(id)


if __name__ == "__main__":
    main()
