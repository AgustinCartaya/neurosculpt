"""Predict the HUNT4 image of the validation patients with a deformation-field method."""
import argparse

from brainpred import splits
from brainpred.methods.dfield import apply, mean, similar_images, similar_images_registered, similar_patches, similar_tissues

STEPS_HELP = """\
steps:
  register_similar  (similar_images_registered only) elastix-register the most similar references
                    to each target and move their longitudinal dfields to the target space
  create_dfields    build the target dfield of each patient from the reference dfields
  apply             warp the HUNT3 T1_MNI / SEG_96 / BMASK_NCRB_MNI with it (--image-types)
Outputs go to <results_dir>/dfield_<method>/{dfields,results}/<id>/...

Defaults reproduce the configuration of the thesis' final comparison:
  mean                       --nb-images 100                       -> *_T1_100
  similar_images             --nb-images 100 --sim-metrics dice_mean -> *_T1_DICE_MEAN_100
  similar_images_registered  --nb-images 5 --reg-types BSP_FAST    -> *_T1_BSP_FAST_5
  similar_tissues(_v2)       --nb-tissues 96 --nb-images 10        -> *_T1_96_10
  similar_patches            --nb-tissues 96 --nb-images 10        -> *_T1_96_10
"""
METHODS = ["mean", "similar_images", "similar_images_registered", "similar_tissues", "similar_tissues_v2", "similar_patches"]
STEPS = ["register_similar", "create_dfields", "apply"]
DEFAULT_NB_IMAGES = {"mean": [100], "similar_images": [100], "similar_images_registered": [5],
                     "similar_tissues": [10], "similar_tissues_v2": [10], "similar_patches": [10]}


def get_ids(args):
    if args.ids:
        return args.ids
    return splits.get_validation_ids()


def run_step(step, args, ids):
    method = args.method
    nb_images_list = args.nb_images or DEFAULT_NB_IMAGES[method]
    apply_kwargs = dict(img_types=args.image_types, skip_existing=args.skip_existing)

    if step == "register_similar":
        if method != "similar_images_registered":
            raise SystemExit("register_similar is only available for --method similar_images_registered")
        for reg_type in args.reg_types:
            for nb_images in nb_images_list:
                similar_images_registered.register_similar_images(ids, nb_images=nb_images, reg_type=reg_type,
                                                                  save_registered_images=args.save_registered_images)
        return

    create = step == "create_dfields"
    if method == "mean":
        if create:
            mean.create_dfields(nb_images_list)
        else:
            mean.apply_dfields(ids, nb_images_list, **apply_kwargs)
    elif method == "similar_images":
        if create:
            similar_images.create_dfields(ids, nb_images_list, args.sim_metrics)
        else:
            similar_images.apply_dfields(ids, nb_images_list, args.sim_metrics, **apply_kwargs)
    elif method == "similar_images_registered":
        if create:
            similar_images_registered.create_dfields(ids, nb_images_list, args.reg_types)
        else:
            similar_images_registered.apply_dfields(ids, nb_images_list, args.reg_types, **apply_kwargs)
    elif method in ("similar_tissues", "similar_tissues_v2"):
        if create:
            similar_tissues.create_dfields(ids, args.nb_tissues, nb_images_list, variant=method)
        else:
            similar_tissues.apply_dfields(ids, args.nb_tissues, nb_images_list, variant=method, **apply_kwargs)
    elif method == "similar_patches":
        if create:
            similar_patches.create_dfields(ids, args.nb_tissues, nb_images_list)
        else:
            similar_patches.apply_dfields(ids, args.nb_tissues, nb_images_list, **apply_kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=STEPS_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--steps", default=None,
                        help=f"comma separated subset of: {','.join(STEPS)} "
                             "(default: create_dfields,apply; plus register_similar for similar_images_registered)")
    parser.add_argument("--ids", nargs="+", help="target patient IDs (default: validation set)")
    parser.add_argument("--nb-images", type=int, nargs="+", help="number of reference patients (see defaults below)")
    parser.add_argument("--nb-tissues", type=int, default=96, choices=[3, 96], help="similar_tissues / similar_patches: segmentation used")
    parser.add_argument("--sim-metrics", nargs="+", default=["dice_mean"], choices=list(similar_images.SIM_METRICS),
                        help="similar_images: similarity used to rank the references")
    parser.add_argument("--reg-types", nargs="+", default=[similar_images_registered.PARAMETERS_FILE_BSPLINE_FAST],
                        choices=list(similar_images_registered.REG_TYPES),
                        help="similar_images_registered: elastix registration of the references to the target")
    parser.add_argument("--save-registered-images", action="store_true",
                        help="similar_images_registered: also save the registered reference T1 / SEG_96 (visual check)")
    parser.add_argument("--image-types", nargs="+", default=list(apply.IMAGE_TYPES), choices=list(apply.IMAGE_TYPES))
    parser.add_argument("--skip-existing", action="store_true", help="apply: do not overwrite existing results")
    args = parser.parse_args()

    if args.steps is None:
        steps = (["register_similar"] if args.method == "similar_images_registered" else []) + ["create_dfields", "apply"]
    else:
        steps = [s.strip() for s in args.steps.split(",") if s.strip()]
        unknown = set(steps) - set(STEPS)
        if unknown:
            parser.error(f"unknown steps: {sorted(unknown)}")

    ids = get_ids(args)
    for step in STEPS:
        if step in steps:
            print(f"==== {args.method}: {step}")
            run_step(step, args, ids)


if __name__ == "__main__":
    main()
