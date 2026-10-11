"""Predict the HUNT4 scan from the preprocessed HUNT3 scan with a trained 3D GAN.

Writes <results_dir>/<method_name>/results/<id>/<id>_0_T1_<checkpoint>.nii.gz, ready for
scripts/postprocess.py and scripts/evaluate.py.

Checkpoints used in the final comparison of the thesis:
    python scripts/predict_gan.py --model mgan      --checkpoints 46   # MGAN_paper_16            (T1_POST_46)
    python scripts/predict_gan.py --model mpgan     --checkpoints 28   # MPGAN_paper_16           (T1_POST_28)
    python scripts/predict_gan.py --model mpgan_seg --checkpoints 52   # MPGAN_paper_16_seg_loss  (T1_RAW_52)
    python scripts/predict_gan.py --model mgan_seg  --checkpoints 65   # MGAN_paper_16_seg_loss   (not in final comparison; no checkpoint selected)

Checkpoint sweep on a single patient (as done during model selection):
    python scripts/predict_gan.py --model mgan --checkpoints 1 9 17 25 33 41 44 46 --ids 08202
"""
import argparse
import os

from brainpred import splits
from brainpred.methods.gan.variants import VARIANTS, configure_tensorflow, get_variant
from brainpred.results import ResultsLayout
from brainpred.utils import io


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--checkpoints", type=int, nargs="+", required=True, help="Checkpoint indices (ckpt-N).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ids", nargs="+", help="Patient IDs to predict.")
    group.add_argument("--split", choices=["validation", "train"], default="validation")
    parser.add_argument("--checkpoints-dir", default=None,
                        help="Override <checkpoints_dir>/<method_name>/training_checkpoints.")
    return parser.parse_args()


def main():
    args = parse_args()
    variant = get_variant(args.model)
    ids = args.ids or (splits.get_validation_ids() if args.split == "validation" else splits.get_train_ids())
    layout = ResultsLayout(variant.method_name)

    configure_tensorflow(variant)
    from brainpred.methods.gan import inference

    for checkpoint in args.checkpoints:
        G = inference.load_generator(variant, checkpoint, checkpoint_dir=args.checkpoints_dir)
        for id in ids:
            print(f"Predicting {id} with checkpoint {checkpoint}")
            result_path_name = layout.get_result_image_path_name(id, "T1", [checkpoint])
            io.create_folders(os.path.dirname(result_path_name))
            img_gen, affine = inference.predict_image(variant, id, G)
            io.save_nifti(img_gen, affine, result_path_name)


if __name__ == "__main__":
    main()
