"""Train one of the 3D GAN variants (MGAN / MPGAN, optionally with segmentation loss).

Checkpoints: <checkpoints_dir>/<method_name>/training_checkpoints/ckpt-N (training resumes from the
latest one). TensorBoard logs: <checkpoints_dir>/<method_name>/logs/.

Models (original folder names):
    mgan       MGAN_paper_16             patch-based cycle GAN (64^3 patches)
    mgan_seg   MGAN_paper_16_seg_loss    + GMM soft-Dice segmentation loss, losses masked by the segmentation
    mpgan      MPGAN_paper_16            whole-volume pix2pix + Models Genesis perceptual loss
    mpgan_seg  MPGAN_paper_16_seg_loss   + GMM soft-Dice segmentation loss

MPGAN needs the Models Genesis weights at <checkpoints_dir>/genesis/Genesis_Chest_CT.h5.
MGAN, MGAN seg and MPGAN need tensorflow_addons (TF <= 2.15).

Examples:
    python scripts/train_gan.py --model mgan
    python scripts/train_gan.py --model mpgan_seg
"""
import argparse

from brainpred.methods.gan import training
from brainpred.methods.gan.variants import VARIANTS, get_variant


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (default: 100000 for MGAN, where an epoch is one patient, 1000 for MPGAN).")
    parser.add_argument("--monitor-each", type=int, default=None,
                        help="Save a checkpoint and TensorBoard summary every N epochs "
                             "(default: mgan 20, mgan_seg 10, mpgan* 1).")
    parser.add_argument("--preview-id", default=None,
                        help="Patient used for the TensorBoard preview (default: first validation patient).")
    parser.add_argument("--show", action="store_true", help="Also display the preview with matplotlib.")
    return parser.parse_args()


def main():
    args = parse_args()
    training.train(get_variant(args.model),
                   batch_size=args.batch_size,
                   num_epochs=args.epochs,
                   monitor_each=args.monitor_each,
                   show=args.show,
                   preview_id=args.preview_id)


if __name__ == "__main__":
    main()
