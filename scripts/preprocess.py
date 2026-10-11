"""Preprocess the HUNT3/HUNT4 T1 scans: SynthSR -> MNI registration -> skull stripping -> normalization -> FastSurfer -> tissues -> mask."""
import argparse

from tqdm import tqdm

from brainpred import paths, splits
from brainpred.preprocessing import cerebellum_mask, fastsurfer, normalization, registration, skull_stripping, synthsr, tissues

STEPS_HELP = """\
steps (comma separated, always executed in this order, for each patient HUNT3 then HUNT4):
  synthsr         1   mri_synthsr on the original (BFC or RAW) T1                -> T1_SYNTHSR
  register_mni    2   mri_robust_register to MNI (HUNT4 to the registered HUNT3) -> T1_MNI + DFIELD_MNI.lta
  register_hunt3  2b  (optional, unused) mri_robust_register HUNT4 -> HUNT3       -> T1_HUNT3
  skull_strip     3   mri_synthstrip on T1_MNI                                   -> BMASK_MNI
  normalize       4   background to 0, scale to [0, 1]                           -> T1_PREP
  segment         5.1 FastSurfer (docker) on T1_PREP                             -> SEG_96/SEG_36/T1_FASTSURFER .mgz in --seg-mgz-folder
  seg_to_patient  5.2 mri_label2vol back to T1_MNI space                         -> .nii.gz in --seg-nii-folder
  copy_seg        5.3 copy the .nii.gz segmentations from --seg-nii-folder to the patient folder
  merge_tissues   6   SEG_36 -> SEG_3
  mask_ncrb       7   brain mask without cerebellum                              -> BMASK_NCRB_MNI

'segment' needs docker + FastSurfer (GPU machine); synthsr/register/skull_strip/seg_to_patient need FreeSurfer.
When those run on different machines, copy the --seg-mgz-folder / --seg-nii-folder folders between them.
"""

STEPS = ["synthsr", "register_mni", "register_hunt3", "skull_strip", "normalize", "segment", "seg_to_patient",
         "copy_seg", "merge_tissues", "mask_ncrb"]


def get_ids(args):
    if args.ids:
        return args.ids
    return {"train": splits.get_train_ids, "validation": splits.get_validation_ids, "all": splits.get_matching_ids}[args.split]()


def preprocess(ids, steps, seg_mgz_folder=None, seg_nii_folder=None, gpu=True):
    progress_bar = tqdm(total=len(ids), desc="Initializing...")
    for id in ids:
        progress_bar.update(1)
        for hunt in splits.HUNTS:
            def step(name):
                if name in steps:
                    progress_bar.set_description(desc=f"{name} HUNT{hunt} {id}")
                    return True
                return False

            if step("synthsr"):
                synthsr.save_synthSR(hunt, id, verify=True)
            if step("register_mni"):
                registration.save_registration_to_MNI_freesurfer(hunt, id)
            if hunt == 4 and step("register_hunt3"):
                registration.save_registration_to_hunt3_freesurfer(id)
            if step("skull_strip"):
                skull_stripping.save_skull_stripping_freesurfer(hunt, id, paths.REGISTRATION_TO_MNI)
            if step("normalize"):
                normalization.save_normalized_image(hunt, id)
            if step("segment"):
                fastsurfer.segment_freesurfer(hunt, id, GPU=gpu, segmentation_folder=seg_mgz_folder, verify=True)
            if step("seg_to_patient"):
                fastsurfer.transform_segmentation_to_patient_space(hunt, id, seg_mgz_folder, seg_nii_folder)
            if step("copy_seg"):
                if seg_nii_folder is None:
                    raise ValueError("copy_seg needs --seg-nii-folder")
                fastsurfer.copy_seg_to_patient_folder(hunt, id, seg_nii_folder)
            if step("merge_tissues"):
                tissues.save_3_tissues_segmentation(hunt, id, verify=False)
            if step("mask_ncrb"):
                cerebellum_mask.create_mask_no_cerebellum(hunt, id)


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=STEPS_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", required=True, help=f"comma separated subset of: {','.join(STEPS)}")
    parser.add_argument("--ids", nargs="+", help="patient IDs (default: the whole --split)")
    parser.add_argument("--split", choices=["train", "validation", "all"], default="all")
    parser.add_argument("--seg-mgz-folder", default=None,
                        help="folder for the FastSurfer .mgz outputs (default: patient folder)")
    parser.add_argument("--seg-nii-folder", default=None,
                        help="folder for the .nii.gz segmentations in patient space (default: patient folder)")
    parser.add_argument("--cpu", action="store_true", help="run FastSurfer with the CPU docker image instead of the GPU one")
    args = parser.parse_args()

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = set(steps) - set(STEPS)
    if unknown:
        parser.error(f"unknown steps: {sorted(unknown)}")

    preprocess(get_ids(args), steps, args.seg_mgz_folder, args.seg_nii_folder, gpu=not args.cpu)


if __name__ == "__main__":
    main()
