"""Per-patient evaluation of predicted HUNT4 images (SSIM, NCC, PSNR, BPF, Dice) against the real HUNT4 scan.

Unifies the per-method ``compute_statistics.py`` / ``compute_statistics_raw.py`` scripts of the
original code. For every validation patient the reference is the real HUNT4 scan registered to MNI,
skull-stripped with its FastSurfer segmentation (SEG_96) and normalised to [0, 1]; its segmentation
is merged to 3 tissues (CSF, GM, WM) with the cerebellum removed. The prediction
``<id>_0_<T1|SEG_96|BMASK_NCRB><type_suffix>_<extra>.nii.gz`` is processed the same way.

With ``raw=True`` (former ``compute_statistics_raw.py``) neither image is skull-stripped or
normalised and only the intensity metrics (ssim, ncc, psnr) are available.

One CSV per metric is written to ``ResultsLayout.get_statistics_path_name(<metric><table_suffix>)``
with one row per patient and one column per ``extra`` (upper-cased).
"""
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from brainpred import paths, splits
from brainpred.utils import brain_metrics as ubm
from brainpred.utils import io, metrics
from brainpred.utils import segmentation as ufs
from brainpred.utils.intensity import skull_stripping_and_normalize

METRICS = ("ssim", "ncc", "psnr", "bpf", "dice")
IMAGE_METRICS = ("ssim", "ncc", "psnr")
DICE_TABLES = ("dice_csf", "dice_gm", "dice_wm")


# ---- ids
def get_available_ids(layout, postprocessed=False):
    """Patients that have a result folder for this method (old ``pil.get_val_ids_2`` for the diffusion model)."""
    folder = os.path.dirname(layout.get_result_patient_folder("x", postprocessed))
    if not os.path.isdir(folder):
        return []
    return sorted(e for e in os.listdir(folder) if os.path.isdir(os.path.join(folder, e)))


def get_ids(split, layout=None, postprocessed=False):
    """``split`` is one of ``validation``, ``train`` or ``available`` (folders present in the method's results)."""
    if split == "validation":
        return splits.get_validation_ids()
    if split == "train":
        return splits.get_train_ids()
    if split == "available":
        return get_available_ids(layout, postprocessed)
    raise ValueError(f"Unknown split: {split}")


# ---- loading
def load_reference(id, raw=False):
    """Real HUNT4 image (MNI) and its 3-tissue segmentation without cerebellum (``None`` if ``raw``)."""
    img_org, _ = io.load_nifti(paths.get_registered_to_template_image_path_name(4, id, paths.REGISTRATION_TO_MNI))
    if raw:
        return img_org, None
    seg_org, _ = io.load_nifti(paths.get_segmentation_path_name(4, id, nb_tissues=96))
    img_org = skull_stripping_and_normalize(img_org, seg_org)
    seg_org = ufs.remove_cerebellum(ufs.merge_seg96_to_seg3(seg_org), seg_org, deep_copy=False)
    return img_org, seg_org


def load_prediction(layout, id, extra, postprocessed=False, type_suffix="", raw=False, load_bmask=False):
    """Predicted image, 3-tissue segmentation without cerebellum and brain mask without cerebellum."""
    extra = (extra,) if isinstance(extra, (str, int)) else tuple(extra)
    img_res, _ = io.load_nifti(layout.get_result_image_path_name(id, "T1" + type_suffix, extra, postprocessed))
    if raw:
        return img_res, None, None
    seg_res, _ = io.load_nifti(layout.get_result_image_path_name(id, "SEG_96" + type_suffix, extra, postprocessed))
    bmask_res = None
    if load_bmask:
        bmask_res, _ = io.load_nifti(layout.get_result_image_path_name(id, "BMASK_NCRB" + type_suffix, extra, postprocessed))
    img_res = skull_stripping_and_normalize(img_res, seg_res)
    seg_res = ufs.remove_cerebellum(ufs.merge_seg96_to_seg3(seg_res), seg_res, deep_copy=False)
    return img_res, seg_res, bmask_res


# ---- computation
def column_name(extra):
    extra = (extra,) if isinstance(extra, (str, int)) else tuple(extra)
    return "_".join(str(e) for e in extra).upper()


def compute_statistics(layout, ids, extras, metrics_list=("ncc", "psnr"), postprocessed=False, type_suffix="",
                       table_suffix="", raw=False, save=True):
    """Compute the requested metrics for every (patient, extra) pair.

    Args:
        layout: :class:`brainpred.results.ResultsLayout` of the evaluated method.
        ids: patient ids.
        extras: list of result ``extra_info`` (str, int or tuple); one table column each.
        metrics_list: subset of :data:`METRICS`.
        postprocessed: read from ``results_postprocessed`` and write to ``statistics_postprocessed``.
        type_suffix: suffix of the image types, e.g. ``"_POST"`` or ``"_RAW"``.
        table_suffix: suffix appended to the CSV names (e.g. ``"_RAW"`` -> ``ncc_RAW.csv``).
        raw: compare un-skull-stripped, un-normalised images (intensity metrics only).
        save: write the CSV files.

    Returns:
        dict ``table_name -> DataFrame`` (index ``id``, one column per extra).
    """
    unknown = set(metrics_list) - set(METRICS)
    if unknown:
        raise ValueError(f"Unknown metrics: {sorted(unknown)}")
    if raw and set(metrics_list) - set(IMAGE_METRICS):
        raise ValueError(f"raw mode only supports {IMAGE_METRICS}")

    columns = [column_name(e) for e in extras]
    n_ids, n_cols = len(ids), len(columns)
    stats = {m: np.zeros((n_ids, n_cols)) for m in metrics_list if m != "dice"}
    if "dice" in metrics_list:
        dices = np.zeros((n_ids, n_cols, 3))

    for i, id in enumerate(tqdm(ids)):
        img_org, seg_org = load_reference(id, raw=raw)
        for j, extra in enumerate(extras):
            img_res, seg_res, bmask_res = load_prediction(layout, id, extra, postprocessed, type_suffix, raw,
                                                          load_bmask="bpf" in metrics_list)
            if "ssim" in stats:
                stats["ssim"][i, j] = metrics.compute_ssim(img_org, img_res)
            if "ncc" in stats:
                stats["ncc"][i, j] = metrics.compute_normalized_cross_correlation(img_org, img_res)
            if "psnr" in stats:
                stats["psnr"][i, j] = metrics.compute_psnr(img_org, img_res)
            if "bpf" in stats:
                stats["bpf"][i, j] = ubm.compute_bpf(seg_res, bmask_res)
            if "dice" in metrics_list:
                dice_res = metrics.compute_multiple_dice_coefficients(seg_org, seg_res, return_dict=False)
                dices[i, j] = dice_res[1:]  # drop background

    tables = {name: _to_df(values, ids, columns) for name, values in stats.items()}
    if "dice" in metrics_list:
        for k, name in enumerate(DICE_TABLES):
            tables[name] = _to_df(dices[..., k], ids, columns)

    if save:
        for name, df in tables.items():
            save_stats(df, layout, name + table_suffix, postprocessed)
    return tables


def _to_df(values, ids, columns):
    df = pd.DataFrame(values, index=ids, columns=columns)
    df.index.name = 'id'
    return df


def save_stats(df, layout, table_name, postprocessed):
    path_name = layout.get_statistics_path_name(table_name, postprocessed=postprocessed)
    io.create_folders(os.path.dirname(path_name))
    df.to_csv(path_name, index=True, header=True)


def load_stats(layout, table_name, postprocessed=False):
    return pd.read_csv(layout.get_statistics_path_name(table_name, postprocessed=postprocessed), index_col='id',
                       dtype={'id': str})


# ---- summary tables (formerly approaches/visualize_statistics.ipynb)
def summarize(tables, initial=None, scale=100):
    """Mean, std and paired t-test against the baseline for each metric table.

    Args:
        tables: dict ``metric -> DataFrame`` (index id, one column per configuration).
        initial: optional dict ``metric -> Series`` (index id) with the baseline value of each
            patient (HUNT3 vs HUNT4, the "initial" row of the thesis tables).
        scale: multiplier applied to every value (100 in the thesis: SSIM and Dice in %).

    Returns:
        dict with DataFrames ``mean``, ``std`` and, when ``initial`` is given, ``t`` and ``p``
        (rows = configurations, columns = metrics).
    """
    from scipy import stats

    out = {"mean": {}, "std": {}, "t": {}, "p": {}}
    for metric, df in tables.items():
        df = df * scale
        out["mean"][metric] = df.mean()
        out["std"][metric] = df.std()
        if initial is not None and metric in initial:
            base = initial[metric].loc[df.index] * scale
            t, p = stats.ttest_rel(df.values, base.values[:, None])
            out["t"][metric] = pd.Series(t, index=df.columns)
            out["p"][metric] = pd.Series(p, index=df.columns)
    return {k: pd.DataFrame(v) for k, v in out.items() if v}
