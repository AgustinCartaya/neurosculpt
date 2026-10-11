"""The four final GAN variants, their hyper-parameters, and where their checkpoints and logs live.

Each variant keeps its original folder name (``method_name``), used both for predictions
(``<results_dir>/<method_name>/results``) and checkpoints
(``<checkpoints_dir>/<method_name>/training_checkpoints/ckpt-N``).
"""
from dataclasses import dataclass

from brainpred import paths
from brainpred.methods.gan.layers.normalization import NORM_GROUP, NORM_TFA

FAMILY_MGAN = "mgan"
FAMILY_MPGAN = "mpgan"


@dataclass(frozen=True)
class GanVariant:
    key: str                 # --model value
    method_name: str         # results / checkpoints folder name
    family: str              # FAMILY_MGAN (3D patches, cycle GAN) or FAMILY_MPGAN (whole volume, pix2pix)
    norm: str                # instance-normalization implementation, see layers/normalization.py
    seg_loss: bool           # adds the GMM-based soft Dice segmentation loss
    final_checkpoint: int | None = None   # checkpoint used in the final comparison of the thesis
    final_result_type: str | None = None  # result type used in that comparison (T1_POST / T1_RAW)
    # ---- training hyper-parameters (as in the original train_GAN.py)
    learning_rate: float = 1e-5
    lr_patience: int = 20
    num_epochs: int = 100000
    monitor_each: int = 20
    shuffle_patches: bool = False


VARIANTS = {
    "mgan": GanVariant("mgan", "MGAN_paper_16", FAMILY_MGAN, NORM_TFA, seg_loss=False,
                       final_checkpoint=46, final_result_type="T1_POST",
                       monitor_each=20, shuffle_patches=False),
    "mgan_seg": GanVariant("mgan_seg", "MGAN_paper_16_seg_loss", FAMILY_MGAN, NORM_TFA, seg_loss=True,
                           monitor_each=10, shuffle_patches=True),
    "mpgan": GanVariant("mpgan", "MPGAN_paper_16", FAMILY_MPGAN, NORM_TFA, seg_loss=False,
                        final_checkpoint=28, final_result_type="T1_POST",
                        learning_rate=1e-4, lr_patience=10, num_epochs=1000, monitor_each=1),
    "mpgan_seg": GanVariant("mpgan_seg", "MPGAN_paper_16_seg_loss", FAMILY_MPGAN, NORM_GROUP, seg_loss=True,
                            final_checkpoint=52, final_result_type="T1_RAW",
                            learning_rate=1e-4, lr_patience=10, num_epochs=1000, monitor_each=1),
}

# ---- data / geometry shared by every variant of a family
IMAGE_SIZE = (193, 229, 193)       # preprocessed MNI-space volumes
MGAN_PATCH_SIZE = (64, 64, 64)
MGAN_PATCH_STEP = (32, 32, 32)
MGAN_BACKGROUND_TH = 0.15          # min fraction of brain voxels for a patch to be used
MPGAN_INPUT_SHAPE = (192, 224, 192)  # volumes are cropped to a size divisible by 2**3
MPGAN_IMG_OFFSET = (0, 0, 0)


def get_variant(key):
    if key not in VARIANTS:
        raise ValueError(f"Unknown model '{key}', choose one of {sorted(VARIANTS)}")
    return VARIANTS[key]


def get_checkpoint_dir(variant):
    return paths.get_checkpoints_folder(variant.method_name, "training_checkpoints")


def get_log_dir(variant):
    return paths.get_checkpoints_folder(variant.method_name, "logs")


def configure_tensorflow(variant):
    """Global TF settings the models were trained with; call once before building any model.

    Mixed precision (float16 compute) for every variant, and channels-first data for MPGAN.
    """
    import tensorflow as tf
    from tensorflow.keras import mixed_precision

    if variant.family == FAMILY_MPGAN:
        tf.keras.backend.set_image_data_format("channels_first")
    mixed_precision.set_global_policy('mixed_float16')
