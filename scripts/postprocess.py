"""Postprocess the predicted HUNT4 images of a method (SynthSR, skull stripping, normalization, FastSurfer, mask without cerebellum)."""
import argparse

from brainpred import splits
from brainpred.postprocessing import pipeline
from brainpred.results import ResultsLayout

EPILOG = """\
Steps (see brainpred.postprocessing.pipeline): copy, register_back, synthsr, skull_strip, normalize,
segment, seg_to_patient, mask_ncrb. 'segment' needs docker + FastSurfer, the other external steps FreeSurfer.

Settings reproducing the original approaches (--steps lists every step that was run, one at a time):
  MGAN_paper_16            --method MGAN_paper_16 --extra 46 --steps synthsr,skull_strip,normalize,segment,seg_to_patient,mask_ncrb
  MPGAN_paper_16           --method MPGAN_paper_16 --extra 28 (same steps)
  MPGAN_paper_16_seg_loss  --method MPGAN_paper_16_seg_loss --extra 52 --source copy --steps copy,synthsr,skull_strip,normalize,segment,seg_to_patient,mask_ncrb
                           then: --seg-input copy --seg-name SEG_96_RAW --steps segment,seg_to_patient,mask_ncrb
  diffusion_model          --method diffusion_model --extra RES --source reg --steps register_back,synthsr,skull_strip,normalize,segment,seg_to_patient,mask_ncrb
  StarGAN_V2_sagital       --method StarGAN_V2_sagital --extra BEST --source reg --no-synthsr --steps register_back,skull_strip,normalize,segment,seg_to_patient,mask_ncrb
  dfield_similar_patches   --method dfield_similar_patches --extra 96 10 --source copy --no-synthsr --seg-name SEG_96 --steps copy,skull_strip,normalize,segment,seg_to_patient,mask_ncrb
  dfield_similar_images    --method dfield_similar_images --extra DICE_MEAN 100 --copy-as T1 --seg-input copy --seg-name SEG_96 --steps copy,segment,seg_to_patient
  0MNI_Segmentation        --image <MNI_T1.nii> --seg-out <MNI_SEG_96.nii.gz> --sid MNI --steps segment,seg_to_patient
  MGAN paper checkpoints   --mgan-paper-folder <first_results> --ids 08202 --checkpoints 17 19 23 25 27 29 21 31 --steps synthsr,skull_strip,hist_match
"""


def get_ids(args):
    if args.ids:
        return args.ids
    return {"train": splits.get_train_ids, "validation": splits.get_validation_ids, "all": splits.get_matching_ids}[args.split]()


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", required=True, help="comma separated list of steps")
    parser.add_argument("--method", help="results folder name of the method (<results_dir>/<method>)")
    parser.add_argument("--extra", nargs="+", default=[], help="extra_info of the result names, e.g. checkpoint '46'")
    parser.add_argument("--ids", nargs="+", help="patient IDs (default: the whole --split)")
    parser.add_argument("--split", choices=["train", "validation", "all"], default="validation")
    parser.add_argument("--source", choices=["raw", "copy", "reg"], default="raw",
                        help="image the chain starts from: raw prediction T1, the copied image, or T1_REG (default: raw)")
    parser.add_argument("--copy-as", default="T1_RAW", help="type name of the copied prediction (default: T1_RAW)")
    parser.add_argument("--no-synthsr", action="store_true", help="skull strip / normalize the source image instead of T1_SYNTHSR")
    parser.add_argument("--seg-input", choices=["post", "copy"], default="post",
                        help="image to segment: T1_POST or the copied image (default: post)")
    parser.add_argument("--seg-name", default="SEG_96_POST",
                        help="type name of the segmentation; the mask without cerebellum replaces SEG_96 by BMASK_NCRB (default: SEG_96_POST)")
    parser.add_argument("--cpu", action="store_true", help="run FastSurfer with the CPU docker image")
    # single image mode
    parser.add_argument("--image", help="segment a single image instead of a method's results")
    parser.add_argument("--seg-out", help="output segmentation (.nii.gz) for --image")
    parser.add_argument("--sid", default="MNI", help="FastSurfer subject id for --image")
    # legacy MGAN paper checkpoints mode
    parser.add_argument("--mgan-paper-folder", help="folder with <id>_CHECK_<checkpoint>.nii.gz predictions")
    parser.add_argument("--checkpoints", nargs="+", default=[17, 19, 23, 25, 27, 29, 21, 31])
    args = parser.parse_args()

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    gpu = not args.cpu

    if args.image:
        if not args.seg_out:
            parser.error("--image needs --seg-out")
        pipeline.segment_single_image(args.sid, args.image, args.seg_out, steps=steps, gpu=gpu)
    elif args.mgan_paper_folder:
        unknown = set(steps) - set(pipeline.MGAN_PAPER_STEPS)
        if unknown:
            parser.error(f"unknown steps: {sorted(unknown)}")
        pipeline.postprocess_mgan_paper_checkpoints(args.mgan_paper_folder, args.ids or ["08202"], args.checkpoints, steps)
    else:
        if not args.method:
            parser.error("--method is required")
        unknown = set(steps) - set(pipeline.STEPS)
        if unknown:
            parser.error(f"unknown steps: {sorted(unknown)}")
        pipeline.postprocess(ResultsLayout(args.method), get_ids(args), tuple(args.extra), steps,
                             source=args.source, copy_as=args.copy_as, use_synthsr=not args.no_synthsr,
                             seg_input=args.seg_input, seg_name=args.seg_name, gpu=gpu)


if __name__ == "__main__":
    main()
