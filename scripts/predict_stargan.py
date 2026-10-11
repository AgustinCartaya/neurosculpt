"""Predict HUNT4 volumes of the validation patients with the trained 2D StarGAN v2 (sagittal slices).

The thesis results use the checkpoint at iteration 65200 and the style of the most similar
training patient:

    python scripts/predict_stargan.py --step 65200 --strategy best

Results: <results_dir>/StarGAN_V2_sagital/results/<id>/<id>_0_T1_<extra>.nii.gz (in PREP_MNI space;
postprocessing brings them back to the patient space). Checkpoints are read from
<checkpoints_dir>/StarGAN_V2_sagital/<step>_nets_ema.ckpt.
"""
import argparse

import torch
from torch.backends import cudnn

from brainpred import splits
from brainpred.methods.stargan_v2 import predict


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", type=int, default=65200, help="Training iteration of the checkpoint.")
    parser.add_argument("--strategy", default="best", choices=sorted(predict.STRATEGIES) + ["custom"],
                        help="How the style (reference) image is chosen.")
    parser.add_argument("--style-id", default="00039", help="Style patient for --strategy custom.")
    parser.add_argument("--no-prep-mni", action="store_true",
                        help="Use the preprocessed scans instead of the PREP_MNI volumes the model was trained on.")
    parser.add_argument("--ids", nargs="+", default=None, help="Patient ids (default: validation split).")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


def main():
    args = parse_args()
    print("CUDA available:", torch.cuda.is_available())
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    ids = args.ids or splits.get_validation_ids()
    prep_mni = not args.no_prep_mni
    nets_ema = predict.load_networks(args.step, args.checkpoint_dir)
    if args.strategy == "custom":
        predict.save_predictions_custom_id(nets_ema, ids, prep_mni, args.style_id)
    else:
        predict.STRATEGIES[args.strategy](nets_ema, ids, prep_mni)


if __name__ == "__main__":
    main()
