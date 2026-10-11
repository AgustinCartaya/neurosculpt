"""3D prediction with the 2D StarGAN v2: every sagittal slice of the HUNT3 scan is translated to the HUNT4 domain.

The style of each slice is taken from the same slice of a reference ("style") HUNT4 scan, chosen as
the patient itself (GT, upper bound), the most similar training patient (tissue Dice similarity,
the configuration reported in the thesis: ``BEST``), the training patient with the highest/lowest
BPF, or a fixed id. Translated slices are un-padded and stacked back into a volume.

Derived from StarGAN v2 (https://github.com/clovaai/stargan-v2), CC BY-NC 4.0, see LICENSE.
"""
import os

import numpy as np
import torch
from munch import Munch
from tqdm import tqdm

from brainpred import paths, splits
from brainpred.data.stargan_2d import DataLoaderImagesStarGAN2DPredict, InputFetcherPredict
from brainpred.methods.stargan_v2.checkpoint import CheckpointIO
from brainpred.methods.stargan_v2.model import build_model
from brainpred.results import ResultsLayout
from brainpred.utils import io
from brainpred.utils.arrays import remove_padding

METHOD_NAME = "StarGAN_V2_sagital"
# Architecture of the trained model (must match scripts/train_stargan.py).
MODEL_ARGS = dict(img_size=256, style_dim=64, w_hpf=1, n_channels=1, latent_dim=16, num_domains=2)
HUNT3 = DataLoaderImagesStarGAN2DPredict.HUNT3
HUNT4 = DataLoaderImagesStarGAN2DPredict.HUNT4


def get_results_layout(prep_mni=True):
    """Predictions on the PREP_MNI volumes (default) or on the preprocessed scans (``..._NO_PREP_MNI``)."""
    return ResultsLayout(METHOD_NAME if prep_mni else f"{METHOD_NAME}_NO_PREP_MNI")


def get_checkpoint_dir():
    return paths.get_checkpoints_folder(METHOD_NAME)


def load_networks(step, checkpoint_dir=None):
    """EMA networks of the checkpoint at iteration ``step``."""
    _, nets_ema = build_model(Munch(MODEL_ARGS))
    ckptio = CheckpointIO(os.path.join(checkpoint_dir or get_checkpoint_dir(), '{:06d}_nets_ema.ckpt'.format(step)),
                          data_parallel=True, **nets_ema)
    ckptio.load(step)
    return nets_ema


def create_fetcher(img_size, img_src, img_sty, bmask_src, bmask_sty, hunt_src, hunt_sty):
    loader = DataLoaderImagesStarGAN2DPredict(img_size=(img_size, img_size), img_src=img_src, img_sty=img_sty,
                                              mask_src=bmask_src, mask_sty=bmask_sty, hunt_src=hunt_src, hunt_sty=hunt_sty,
                                              axis=0, channels_first=True)
    return InputFetcherPredict(loader)


@torch.no_grad()
def predict3D(style_encoder, generator, fetcher_pred):
    def denormalize(x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    predicted_layers = []
    for inputs in tqdm(fetcher_pred):
        x_real, mask_real = inputs.x_src, inputs.mask
        x_ref, y_ref = inputs.x_ref, inputs.y_ref

        s_ref = style_encoder(x_ref, y_ref)
        x_fake = denormalize(generator(x_real, s_ref, masks=mask_real))
        predicted_layers.append(np.squeeze(x_fake.cpu().numpy()))
    return predicted_layers


def load_images(id_src, id_sty, hunt_src, hunt_sty, prep_mni):
    """Source and style volumes with their brain masks (hunt_* are 0 for HUNT3, 1 for HUNT4)."""
    if prep_mni:
        from brainpred.preprocessing import slices_2d
        img_src, affine = io.load_nifti(slices_2d.get_registered_image_path_name(hunt_src + 3, id_src))
        bmask_src = io.load_nifti(slices_2d.get_registered_bmask_path_name(hunt_src + 3, id_src))[0]
        img_sty = io.load_nifti(slices_2d.get_registered_image_path_name(hunt_sty + 3, id_sty))[0]
        bmask_sty = io.load_nifti(slices_2d.get_registered_bmask_path_name(hunt_sty + 3, id_sty))[0]
    else:
        img_src, affine = io.load_nifti(paths.get_preprocessed_image_path_name(hunt_src + 3, id_src))
        bmask_src = io.load_nifti(paths.get_brain_mask_path_name(hunt_src + 3, id_src, paths.REGISTRATION_TO_MNI))[0]
        img_sty = io.load_nifti(paths.get_preprocessed_image_path_name(hunt_sty + 3, id_sty))[0]
        bmask_sty = io.load_nifti(paths.get_brain_mask_path_name(hunt_sty + 3, id_sty, paths.REGISTRATION_TO_MNI))[0]
    return img_src, img_sty, bmask_src, bmask_sty, affine


def load_and_predict(nets_ema, id_src, id_sty, prep_mni, hunt_sty=HUNT4):
    """Translate the HUNT3 scan of ``id_src`` using the style of scan ``hunt_sty`` of ``id_sty``."""
    img_src, img_sty, bmask_src, bmask_sty, affine = load_images(id_src, id_sty, HUNT3, hunt_sty, prep_mni=prep_mni)
    fetcher_pred = create_fetcher(MODEL_ARGS["img_size"], img_src, img_sty, bmask_src, bmask_sty, HUNT3, hunt_sty)
    predicted_layers = predict3D(nets_ema.style_encoder, nets_ema.generator, fetcher_pred)
    predicted_layers = [remove_padding(layer, img_src.shape[1:]) for layer in predicted_layers]
    return np.stack(predicted_layers, axis=0), affine


def _predict_and_save(nets_ema, id, id_sty, extra_info, prep_mni, hunt_sty=HUNT4):
    result_path_name = get_results_layout(prep_mni).get_result_image_path_name(id, "T1", extra_info)
    if io.verify_file(result_path_name):
        return
    if io.verify_folder(os.path.dirname(result_path_name), create=True):
        img_pred, affine = load_and_predict(nets_ema, id, id_sty, prep_mni, hunt_sty=hunt_sty)
        io.save_nifti(img_pred, affine, result_path_name)


# ---- style selection strategies (result extra in brackets)
def save_predictions_using_gt(nets_ema, ids, prep_mni):
    """[GT] style = the patient's own HUNT4 scan (upper bound)."""
    for id in ids:
        print(f"predicting {id} using {id}")
        _predict_and_save(nets_ema, id, id, ["GT"], prep_mni)


def save_predictions_using_most_similar_image(nets_ema, ids, prep_mni):
    """[BEST] style = HUNT4 scan of the most similar patient (HUNT3 tissue ``dice_mean``), excluding ``ids``."""
    from brainpred.features.inter_patient.tissue_based import FeaturesLoader

    dice_mean_table_reader = FeaturesLoader(3, splits.get_matching_ids())
    for id in ids:
        df_patient_sim_table = dice_mean_table_reader.get_sorted_similarity_table(
            id, "dice_mean", 1, exclude_ids=list(ids) + splits.get_excluded_patients())
        id_sty = df_patient_sim_table.iloc[0]['id']
        print(f"predicting {id} using {id_sty}")
        _predict_and_save(nets_ema, id, id_sty, ["BEST"], prep_mni)


def save_predictions_using_differents_bpf(nets_ema, ids, prep_mni, bpf_type="HIGH"):
    """[<id_sty>_HIGH|LOW] style = patient with the highest / lowest HUNT3 BPF, excluding ``ids``."""
    from brainpred.features.cross_sectional.whole_brain import FeaturesLoader

    bpf_table = FeaturesLoader().get_df(columns=["bpf_hunt3"], remove_columns=False, return_patient_id=True,
                                        id_patients=list(ids) + splits.get_excluded_patients(), remove_patients=True)
    if bpf_type == "HIGH":
        bpf_index = bpf_table['bpf_hunt3'].idxmax()
    elif bpf_type == "LOW":
        bpf_index = bpf_table['bpf_hunt3'].idxmin()
    else:
        raise ValueError(f"Unknown bpf_type: {bpf_type}")
    id_sty = f"{int(bpf_table.at[bpf_index, 'patient_id']):05d}"

    for id in ids:
        print(f"predicting {id} using {id_sty} BPF: {bpf_type}")
        _predict_and_save(nets_ema, id, id_sty, [id_sty, bpf_type], prep_mni)


def save_predictions_using_self(nets_ema, ids, prep_mni, hunt=HUNT4):
    """[<id>_SELF_STYLE_HUNT<h>] style = the patient's own scan of visit ``hunt``."""
    for id in ids:
        print(f"predicting {id} using {id} SELF")
        _predict_and_save(nets_ema, id, id, [id, "SELF", f"STYLE_HUNT{hunt + 3}"], prep_mni, hunt_sty=hunt)


def save_predictions_custom_id(nets_ema, ids, prep_mni, id_sty):
    """[<id_sty>_CUSTOM] style = HUNT4 scan of a fixed patient."""
    for id in ids:
        print(f"predicting {id} using CUSTOM {id_sty}")
        _predict_and_save(nets_ema, id, id_sty, [id_sty, "CUSTOM"], prep_mni)


STRATEGIES = {
    "gt": save_predictions_using_gt,
    "best": save_predictions_using_most_similar_image,
    "bpf_high": lambda nets, ids, prep_mni: save_predictions_using_differents_bpf(nets, ids, prep_mni, "HIGH"),
    "bpf_low": lambda nets, ids, prep_mni: save_predictions_using_differents_bpf(nets, ids, prep_mni, "LOW"),
    "self_hunt3": lambda nets, ids, prep_mni: save_predictions_using_self(nets, ids, prep_mni, hunt=HUNT3),
    "self_hunt4": lambda nets, ids, prep_mni: save_predictions_using_self(nets, ids, prep_mni, hunt=HUNT4),
}
