"""Qualitative figures: per-method result viewers and the cross-method comparison figures of the thesis.

Ports ``approaches/*/visualize_results.py`` (per method), ``approaches/visualize_results.py``
(comparison of all methods) and ``approaches/visualize_results_poster.py`` (single-slice and
deformation-field figures of the final presentation / poster). Interactive viewers use
:func:`brainpred.utils.display.imgshow_3D_list`; saved figures go to ``paths.get_figures_folder()``.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from tqdm import tqdm

from brainpred import paths, splits
from brainpred.results import ResultsLayout
from brainpred.utils import display, io
from brainpred.utils import segmentation as ufs
from brainpred.utils.intensity import normalize, skull_stripping_and_normalize

DPI = 1000

# FreeSurfer labels shown in the "important tissues" figures, as (right, left) pairs, and their colours.
IMPORTANT_TISSUES = [42, 3, 43, 4, 53, 17, 49, 10, 2006, 1006, 2009, 1009, 2029, 1029, 2027, 1027, 2028, 1028, 2025, 1025, 2005, 1005]
TISSUE_COLORS = {
    "red": (255, 0, 0),          # cortex (R:42, L:3)
    "purple": (128, 0, 128),     # lateral ventricles (R:43, L:4)
    "yellow": (225, 225, 0),     # hippocampus (R:53, L:17)
    "blue": (0, 0, 255),         # thalamus (R:49, L:10)
    "green": (0, 255, 0),        # entorhinal cortex (R:2006, L:1006)
    "cyan": (0, 255, 255),       # inferior temporal gyri (R:2009, L:1009)
    "magenta": (255, 0, 255),    # superior parietal (R:2029, L:1029)
    "orange": (255, 165, 0),     # middle frontal gyrus (R:2027, L:1027)
    "pink": (255, 192, 203),     # superior frontal gyrus (R:2028, L:1028)
    "lime": (10, 128, 90),       # precuneus (R:2025, L:1025)
    "brown": (165, 42, 42),      # cuneus (R:2005, L:1005)
}

# Example validation patients used in the thesis figures (index into splits.get_validation_ids()).
CASES = {
    "LOW": 78,     # 17112  axial(2)=96, sagittal(0)=77, coronal(1)=122
    "MEDIUM": 80,  # 17310  axial=96, sagittal=75, coronal=122 (the per-method viewers used index 41 = 14968)
    "HIGH": 48,    # 15170  axial=103, sagittal=80, coronal=122
}
# Most similar training patients to 17112 (used in the poster deformation-field figures).
MOST_SIMILAR_TO_17112 = ["13329", "12582", "12742"]

# ---------------------------------------------------------------------------------------------
# Method -> result file used in the comparison figures.
#   method:        folder in <results_dir> (ResultsLayout name)
#   extra:         result "extra" string (hyper-parameters joined with "_")
#   type_suffix:   image type suffix ("" raw prediction, "_POST" postprocessed, "_RAW" seg-loss raw)
#   postprocessed: read from results_postprocessed/
# Files: <results_dir>/<method>/results[_postprocessed]/<id>/<id>_0_{T1,SEG_96}<type_suffix>_<extra>.nii.gz
# ---------------------------------------------------------------------------------------------
RESULT_SOURCES = {
    "MEAN": dict(method="dfield_mean", extra="100"),
    "SIM_IMG": dict(method="dfield_similar_images", extra="DICE_MEAN_100"),
    "SIM_IMG_REG": dict(method="dfield_similar_images_registered", extra="BSP_FAST_5"),
    "SIM_TIS": dict(method="dfield_similar_tissues", extra="96_10"),
    "SIM_PATCH": dict(method="dfield_similar_patches", extra="96_10"),
    "SIM_PATCH_POST": dict(method="dfield_similar_patches", extra="96_10", type_suffix="_POST", postprocessed=True),
    "MPGAN": dict(method="MPGAN_paper_16", extra="28", type_suffix="_POST", postprocessed=True),
    "MPGAN_SEG": dict(method="MPGAN_paper_16_seg_loss", extra="52", type_suffix="_RAW", postprocessed=True),
    "MGAN": dict(method="MGAN_paper_16", extra="46", type_suffix="_POST", postprocessed=True),
    "StyleGAN": dict(method="StarGAN_V2_sagital", extra="BEST", type_suffix="_POST", postprocessed=True),
    "DIFFUSION": dict(method="diffusion_model", extra="RES", type_suffix="_POST", postprocessed=True),
}

# Predicted deformation fields shown in the poster figures (<results_dir>/<method>/dfields/<id>/<pattern>).
# Note: SIM_TIS uses the 100-image field while its image result above uses 10 images (as in the original).
DFIELD_SOURCES = {
    "SIM_IMG": dict(method="dfield_similar_images", pattern="{0}_IMGSIM_DFIELD_{1}_{2}.nii.gz", extra=("DICE_MEAN", 100)),
    "SIM_IMG_REG": dict(method="dfield_similar_images_registered", pattern="{0}_IMGSIM_DFIELD_{1}_{2}.nii.gz", extra=("BSP_FAST", 5)),
    "SIM_TIS": dict(method="dfield_similar_tissues", pattern="{0}_TISSUESIM_DFIELD_{1}_{2}.nii.gz", extra=(96, 100)),
    "SIM_PATCH": dict(method="dfield_similar_patches", pattern="{0}_PATCHSIM_DFIELD_{1}_{2}.nii.gz", extra=(96, 10)),
}

FAMILIES = {
    "DF": ["HUNT3", "HUNT4", "REG", "SIM_IMG", "SIM_IMG_REG", "SIM_TIS", "SIM_PATCH", "SIM_PATCH_POST"],
    "GAN": ["HUNT3", "HUNT4", "MPGAN", "MPGAN_SEG", "MGAN", "StyleGAN"],
    "BEST": ["HUNT3", "HUNT4", "SIM_PATCH_POST", "MPGAN_SEG"],
    "MNI": ["MNI"],
    "MEAN": ["HUNT3", "HUNT4", "MEAN"],
    "ONLY_DF": ["REG", "SIM_IMG", "SIM_IMG_REG", "SIM_TIS", "SIM_PATCH"],
}


def get_case_id(case):
    """Validation id for ``LOW``/``MEDIUM``/``HIGH``, or ``case`` itself if it is already an id."""
    if case in CASES:
        return splits.get_validation_ids()[CASES[case]]
    return case


def get_figure_path_name(name):
    folder = paths.get_figures_folder()
    io.create_folders(folder)
    return f"{folder}/{name}.png"


def _save_dict(save, name, dpi=DPI):
    return {"dpi": dpi, "path_name": get_figure_path_name(name)} if save else None


# ---- paths
def similar_images_registration_path_name(id, type_img, target_id="17112"):
    """Training image ``id`` registered (BSP_FAST) to validation patient ``target_id`` by dfield_similar_images_registered."""
    folder = f"{ResultsLayout('dfield_similar_images_registered').get_images_path()}/similar_images_registrations/{target_id}"
    return f"{folder}/{target_id}_{id}_{type_img}_BSP_FAST.nii.gz"


def get_image_path_names(id, name):
    """(T1 path, SEG_96 path) of one entry of a comparison figure."""
    if name in ("HUNT3", "HUNT4"):
        hunt = int(name[-1])
        return paths.get_preprocessed_image_path_name(hunt, id), paths.get_segmentation_path_name(hunt, id, 96)
    if name == "REG":
        return (paths.get_longitudinal_registration_path_name(id, "T1"),
                paths.get_longitudinal_registration_path_name(id, "SEG_96"))
    if name == "MNI":
        folder = ResultsLayout("0MNI_Segmentation").get_images_path()
        return f"{folder}/MNI_T1.nii.gz", f"{folder}/MNI_SEG_96.nii.gz"
    if name == "SIM_IMG_REG_REG":
        return similar_images_registration_path_name(id, "T1"), similar_images_registration_path_name(id, "SEG_96")
    src = RESULT_SOURCES[name]
    layout = ResultsLayout(src["method"])
    suffix, post = src.get("type_suffix", ""), src.get("postprocessed", False)
    return (layout.get_result_image_path_name(id, "T1" + suffix, (src["extra"],), post),
            layout.get_result_image_path_name(id, "SEG_96" + suffix, (src["extra"],), post))


def get_dfield_figure_path_names(id, name):
    """(background T1, SEG_96, deformation field) for the poster deformation-field figures.

    REG and SIM_IMG_REG_REG use their own registered image as background; the predicted fields of
    the deformation-field methods are drawn over the patient's HUNT3 image.
    """
    if name == "REG":
        img, seg = get_image_path_names(id, "REG")
        return img, seg, paths.get_longitudinal_registration_path_name(id, "DFIELD")
    if name == "SIM_IMG_REG_REG":
        img, seg = get_image_path_names(id, name)
        return img, seg, similar_images_registration_path_name(id, "DFIELD")
    src = DFIELD_SOURCES[name]
    dfield = ResultsLayout(src["method"], src["pattern"]).get_dfield_path_name(id, *src["extra"])
    img, seg = get_image_path_names(id, "HUNT3")
    return img, seg, dfield


def _family_names(family):
    return list(family) if isinstance(family, (list, tuple)) else FAMILIES[family]


def _load_ss(img_path_name, seg_path_name):
    img = io.load_nifti(img_path_name)[0]
    seg = io.load_nifti(seg_path_name)[0]
    return skull_stripping_and_normalize(img, seg), seg


# =============================================================================================
# Comparison of all methods (approaches/visualize_results.py)
# =============================================================================================
def visualize_HUNT_SEG_images(ids, hunt=4, nb_tissues=3, mooving_plane=2, current_plane=110):
    nb_ids = len(ids)
    img_list = [None] * nb_ids * 2
    img_names = [None] * nb_ids * 2
    for i, id in enumerate(tqdm(ids)):
        img = io.load_nifti(paths.get_preprocessed_image_path_name(hunt, id))[0]
        seg = io.load_nifti(paths.get_segmentation_path_name(hunt, id, nb_tissues))[0]
        if nb_tissues > 3:
            img = np.stack([img, img, img], axis=3)
            seg = ufs.seg_to_rgbseg(seg) / 255
        else:
            seg /= nb_tissues
        img_list[i], img_list[i + nb_ids] = img, seg
        img_names[i], img_names[i + nb_ids] = f"{id}", f"{id}"
    display.imgshow_3D_list(f"HUNT4 and SEG {nb_tissues}", img_list, subimg_titles=img_names, subtitle_help=(20, 10),
                            custom_disposition=(2, -1), subtitle_space=10, mooving_plane=mooving_plane, current_plane=current_plane)


def visualize_HUNT3_HUNT4_images(ids, mooving_plane=2, current_plane=110):
    nb_ids = len(ids)
    img_list = [None] * nb_ids * 2
    img_names = [None] * nb_ids * 2
    for i, id in enumerate(tqdm(ids)):
        img_list[i] = io.load_nifti(paths.get_preprocessed_image_path_name(3, id))[0]
        img_list[i + nb_ids] = io.load_nifti(paths.get_preprocessed_image_path_name(4, id))[0]
        img_names[i], img_names[i + nb_ids] = f"{id}", f"{id}"
    display.imgshow_3D_list("HUNT3 vs HUNT4", img_list, subimg_titles=img_names, subtitle_help=(20, 10),
                            custom_disposition=(2, -1), subtitle_space=10, mooving_plane=mooving_plane, current_plane=current_plane)


def visualize_final_results(id, family="DF", mooving_plane=2, current_plane=110, save=False):
    img_names = _family_names(family)
    img_list = [_load_ss(*get_image_path_names(id, name))[0] for name in tqdm(img_names)]
    display.imgshow_3D_list(f"Results ID: {id}", img_list, subimg_titles=img_names, subtitle_help=(0, 0), custom_disposition=(-1, 4),
                            subtitle_space=40, save_dict=_save_dict(save, f"{id}_res_{family}_{mooving_plane}_{current_plane}"),
                            mooving_plane=mooving_plane, current_plane=current_plane)


def visualize_final_results_dif(id, family="DF", mooving_plane=2, current_plane=110, save=False):
    """|prediction - HUNT4| maps (hot colormap)."""
    img_names = _family_names(family)
    img_hunt4 = _load_ss(paths.get_preprocessed_image_path_name(4, id), paths.get_segmentation_path_name(4, id, 96))[0]

    img_list = [np.ones_like(img_hunt4)]  # first tile = 1 so that the hot colormap spans [0, 1]
    for name in tqdm(img_names):
        img = _load_ss(*get_image_path_names(id, name))[0]
        img_list.append(np.abs(img - img_hunt4))

    display.imgshow_3D_list(f"Results ID: {id}", img_list, subimg_titles=["HUNT4"] + img_names, subtitle_help=(0, 0),
                            custom_disposition=(-1, 4), subtitle_space=40, cmap="hot",
                            save_dict=_save_dict(save, f"{id}_dif_{family}_{mooving_plane}_{current_plane}"),
                            mooving_plane=mooving_plane, current_plane=current_plane)


def _merge_seg(seg, nb_tissues):
    if nb_tissues == 3:
        return ufs.merge_seg96_to_seg3(seg)
    if nb_tissues == 36:
        return ufs.merge_seg96_to_seg36(seg)
    return seg


def visualize_final_results_seg_dif(id, nb_tissues=96, see_tissues=None, family="DF", mooving_plane=2, current_plane=110, save=False):
    """Voxels whose label differs from the HUNT4 segmentation."""
    img_names = _family_names(family)
    seg_hunt4 = io.load_nifti(paths.get_segmentation_path_name(4, id, 96))[0]
    if see_tissues is not None:
        seg_hunt4[ufs.get_tissue_mask(seg_hunt4, see_tissues) == 0] = 0
    seg_hunt4 = _merge_seg(seg_hunt4, nb_tissues)

    img_list = [np.zeros_like(seg_hunt4)]
    for name in tqdm(img_names):
        seg = io.load_nifti(get_image_path_names(id, name)[1])[0]
        if see_tissues is not None:
            seg[ufs.get_tissue_mask(seg, see_tissues) == 0] = 0
        seg = _merge_seg(seg, nb_tissues)
        img_list.append(1 - (seg == seg_hunt4).astype(int))

    display.imgshow_3D_list(f"Results ID: {id}", img_list, subimg_titles=["HUNT4"] + img_names, subtitle_help=(0, 0),
                            custom_disposition=(-1, 4), subtitle_space=40,
                            save_dict=_save_dict(save, f"{id}_seg_dif_{family}_{mooving_plane}_{current_plane}"),
                            mooving_plane=mooving_plane, current_plane=current_plane)


def visualize_final_results_seg(id, nb_tissues=96, see_tissues=None, family="DF", mooving_plane=2, current_plane=110, save=False):
    """Images (top row) and their segmentations (bottom row)."""
    img_names = _family_names(family)
    nb_images = len(img_names)
    img_list = [None] * nb_images * 2
    for i, name in enumerate(tqdm(img_names)):
        img, seg = _load_ss(*get_image_path_names(id, name))
        if see_tissues is not None:
            seg[ufs.get_tissue_mask(seg, see_tissues) == 0] = 0

        if nb_tissues > 3:
            if nb_tissues == 36:
                seg = ufs.merge_seg96_to_seg36(seg)
            img = np.stack([img, img, img], axis=3)
            seg = ufs.seg_to_rgbseg(seg) / 255
        else:
            seg = ufs.merge_seg96_to_seg3(seg) / 3
        img_list[i], img_list[i + nb_images] = img, seg

    display.imgshow_3D_list(f"Results ID: {id}", img_list, subimg_titles=img_names + [""] * nb_images, subtitle_help=(0, 0),
                            custom_disposition=(2, -1), subtitle_space=40,
                            save_dict=_save_dict(save, f"{id}_seg_{family}_{mooving_plane}_{current_plane}"),
                            mooving_plane=mooving_plane, current_plane=current_plane)


def important_tissues2rgb(seg, paires_tissues, colors):
    rgb = np.zeros((*seg.shape, 3), dtype=np.float32)
    c = 0
    for i in range(0, len(paires_tissues), 2):
        rgb[seg == paires_tissues[i]] = colors[c]
        rgb[seg == paires_tissues[i + 1]] = colors[c]
        c += 1
    return rgb


def visualize_final_results_seg_important_tissues(id, see_tissues=IMPORTANT_TISSUES, family="DF", mooving_plane=2,
                                                  current_plane=110, save=False):
    """Images and the IMPORTANT_TISSUES coloured over the grey 3-tissue segmentation."""
    img_names = _family_names(family)
    nb_images = len(img_names)
    img_list = [None] * nb_images * 2
    for i, name in enumerate(tqdm(img_names)):
        img, seg96 = _load_ss(*get_image_path_names(id, name))

        seg36 = ufs.merge_seg96_to_seg36(seg96)
        seg3 = ufs.merge_seg96_to_seg3(seg96)

        seg96[ufs.get_tissue_mask(seg96, see_tissues) == 0] = 0
        seg36[ufs.get_tissue_mask(seg36, see_tissues) == 0] = 0

        seg36[seg96 > 0] = seg96[seg96 > 0]
        seg3[seg36 > 0] = 0

        seg_rgb = important_tissues2rgb(seg36, see_tissues, list(TISSUE_COLORS.values())) / 255
        seg3 = np.stack([seg3, seg3, seg3], axis=3) / 3
        seg_rgb = seg_rgb + seg3

        img_list[i], img_list[i + nb_images] = np.stack([img, img, img], axis=3), seg_rgb

    display.imgshow_3D_list(f"Results ID: {id}", img_list, subimg_titles=img_names + [""] * nb_images, subtitle_help=(0, 0),
                            custom_disposition=(2, -1), subtitle_space=40,
                            save_dict=_save_dict(save, f"{id}_segI_{family}_{mooving_plane}_{current_plane}"),
                            mooving_plane=mooving_plane, current_plane=current_plane)


# =============================================================================================
# Poster / presentation figures (approaches/visualize_results_poster.py)
# =============================================================================================
def intercalate_zeros(U, step=2):
    ny, nx = U.shape
    U_new = np.zeros((step * ny, step * nx))
    U_new[::step, ::step] = U  # original values at every ``step``-th position
    return U_new


def closest_divisible(A, B):
    return A - (A % B)


def _quiver_inputs(U, V, bg_img, grid_step):
    div_x, div_y = closest_divisible(bg_img.shape[0], grid_step), closest_divisible(bg_img.shape[1], grid_step)
    bg_img = bg_img[:div_x, :div_y].T
    U = U[:div_x, :div_y].T
    V = V[:div_x, :div_y].T

    ny, nx = U.shape
    X, Y = np.meshgrid(np.linspace(0, nx, nx), np.linspace(0, ny, ny))

    # reduce the grid resolution
    U = intercalate_zeros(U[::grid_step, ::grid_step], step=grid_step)
    V = intercalate_zeros(V[::grid_step, ::grid_step], step=grid_step)

    magnitude = np.sqrt(U**2 + V**2)
    norm_magnitude = magnitude / np.max(magnitude)

    # drop (near) null vectors
    mask = (np.abs(U) > 0.4) | (np.abs(V) > 0.4)
    return X[mask], Y[mask], U[mask], V[mask], norm_magnitude[mask], bg_img, nx, ny


def plot_deformation_field(U, V, bg_img, title="", arrow_scale=0.03, headwidth=2, headlength=3, grid_step=1, width=0.002,
                           save_dict=None, transparent_background=False):
    """Quiver plot of a 2D deformation field over ``bg_img``.

    ``transparent_background=True`` is the former ``plot_deformation_field_alone`` (black pixels transparent).
    """
    X, Y, U, V, norm_magnitude, bg_img, nx, ny = _quiver_inputs(U, V, bg_img, grid_step)

    plt.figure(figsize=(10, 10))
    if transparent_background:
        bg_img_rgba = np.zeros((bg_img.shape[0], bg_img.shape[1], 4))
        bg_img_rgba[..., :3] = bg_img[..., np.newaxis]
        bg_img_rgba[..., 3] = np.where(bg_img == 0, 0, 1)
        plt.imshow(bg_img_rgba, extent=[0, nx, 0, ny], origin='lower', cmap='gray')
    else:
        plt.imshow(bg_img, extent=[0, nx, 0, ny], origin='lower', cmap='gray')
    plt.quiver(X, Y, U, V, norm_magnitude, scale=1 / arrow_scale, cmap='turbo', headwidth=headwidth, headlength=headlength, width=width)
    plt.colorbar(label='Deformation magnitude (normalized)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)

    if save_dict is not None:
        plt.savefig(save_dict["path_name"], dpi=save_dict["dpi"], bbox_inches='tight', transparent=transparent_background)
    plt.show()


def plot_deformation_field_alone(U, V, bg_img, **kwargs):
    plot_deformation_field(U, V, bg_img, transparent_background=True, **kwargs)


def normalize_vectors(vectors):
    magnitudes = np.linalg.norm(vectors, axis=-1, keepdims=True)
    magnitudes[magnitudes == 0] = 1
    return vectors / magnitudes


def calculate_magnitude(vectors):
    return np.sqrt(vectors[..., 0]**2 + vectors[..., 1]**2 + vectors[..., 2]**2)


def get_circular_se(seg_2):
    radio = seg_2 // 2
    element = np.zeros((seg_2, seg_2, seg_2))
    center = seg_2 // 2
    x, y, z = np.ogrid[:seg_2, :seg_2, :seg_2]
    distance = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
    element[np.where(distance <= radio)] = 1
    return element


def dilate_tissue(mask, se=7):
    return ndimage.binary_dilation(mask, structure=get_circular_se(se)).astype(mask.dtype)


def get_tissue_borders(seg, se=7):
    """WM/GM and WM/CSF borders of a 3-tissue segmentation."""
    mask_wm = dilate_tissue(np.where(seg == 3, 1, 0), se=se)
    mask_gm = np.where(seg == 2, 1, 0)
    mask_csf = np.where(seg == 1, 1, 0)
    return mask_wm * mask_gm + mask_wm * mask_csf


def giva_layer_and_magnitude(df):
    """In-plane direction of the field scaled by its 3D magnitude."""
    magnitude = calculate_magnitude(df)
    df_norm = normalize_vectors(df[..., :2])
    df_norm *= magnitude[..., np.newaxis]
    return df_norm


def visualize_final_df(id, nb_tissues=96, see_tissues=None, family=("REG",), extra_info="", only_df=False,
                       layer=113, mooving_plane=2, current_plane=113, save=False):
    """Deformation field (axial ``layer``) of each entry of ``family`` over its masked image."""
    for name in tqdm(_family_names(family)):
        img_path_name, seg_path_name, df_path_name = get_dfield_figure_path_names(id, name)
        img, seg = _load_ss(img_path_name, seg_path_name)
        df = io.load_nifti(df_path_name)[0][..., 0, :]

        seg = _merge_seg(seg, nb_tissues)
        if see_tissues is not None:
            seg[ufs.get_tissue_mask(seg, see_tissues) == 0] = 0

        seg_mask = np.where(seg > 0, 1, 0)
        img = img * seg_mask

        draw_borders = see_tissues is None and extra_info in ["sim_tissue", "sim_tissue_df"]
        if draw_borders:
            mask_borders = get_tissue_borders(seg, se=1)
            img[mask_borders == 1] = 1
            seg_mask[mask_borders == 1] = 0

        for c in range(3):
            df[..., c] = df[..., c] * seg_mask

        df = giva_layer_and_magnitude(df)
        U = df[:, :, layer, 0]
        V = df[:, :, layer, 1]
        img_bg = img[:, :, layer]

        save_dict = _save_dict(save, f"{id}_img_df_{name}_{mooving_plane}_{current_plane}_{extra_info}")
        style = dict(title=f"{name}", arrow_scale=0.019, headwidth=4, headlength=6, grid_step=5, width=0.004, save_dict=save_dict)
        if only_df:
            img_bg = mask_borders[:, :, layer] if draw_borders else np.zeros_like(img_bg)
            plot_deformation_field_alone(U, V, img_bg, **style)
        else:
            plot_deformation_field(U, V, img_bg, **style)


def plot_image_alone(bg_img, title="", save_dict=None):
    plt.imshow(bg_img.T, origin='lower', cmap='gray')
    plt.title(title)
    if save_dict is not None:
        plt.savefig(save_dict["path_name"], dpi=save_dict["dpi"], bbox_inches='tight')
    plt.show()


def visualize_image_alone(id, family=("HUNT3",), extra_info="", mooving_plane=2, current_plane=113, save=False):
    """Single axial slice of each entry (skull-stripped)."""
    for name in tqdm(_family_names(family)):
        img = _load_ss(*get_image_path_names(id, name))[0][:, :, current_plane]
        plot_image_alone(img, title=f"{name}",
                         save_dict=_save_dict(save, f"{id}_img_{name}_{mooving_plane}_{current_plane}_{extra_info}"))


def visualize_tissue_alone(id, see_tissues, nb_tissues=3, family=("HUNT3",), extra_info="", mooving_plane=2, current_plane=113, save=False):
    """Single axial slice of the selected tissue labels."""
    for name in tqdm(_family_names(family)):
        seg = io.load_nifti(get_image_path_names(id, name)[1])[0]
        seg = _merge_seg(seg, nb_tissues)
        seg[ufs.get_tissue_mask(seg, see_tissues) == 0] = 0
        plot_image_alone(seg[:, :, current_plane], title=f"{name}",
                         save_dict=_save_dict(save, f"{id}_seg{see_tissues}_{name}_{mooving_plane}_{current_plane}_{extra_info}"))


# =============================================================================================
# Per-method viewers (approaches/<method>/visualize_results.py)
# =============================================================================================
def load_original_img(id, hunt, skull_stripping=True, normalize_if_not_stripped=False, tissue=None):
    """Real scan registered to MNI and its SEG_96.

    Without skull stripping the deformation-field viewers showed the raw image while the GAN viewers
    normalised it inside the brain (``normalize_if_not_stripped=True``).
    """
    img = io.load_nifti(paths.get_registered_to_template_image_path_name(hunt, id, paths.REGISTRATION_TO_MNI))[0]
    seg = io.load_nifti(paths.get_segmentation_path_name(hunt, id, 96))[0]
    if skull_stripping:
        img = skull_stripping_and_normalize(img, seg)
        img, seg = apply_tissue_mask(img, seg, tissue)
    elif normalize_if_not_stripped:
        img = normalize(img, seg)
    return img, seg


def apply_tissue_mask(img, seg, tissue):
    """Keep only ``tissue`` (a SEG_96 label or CSF/GM/WM); returns the image and the binary mask."""
    if tissue is None:
        return img, seg
    if tissue in ["CSF", "GM", "WM"]:
        seg = ufs.merge_seg96_to_seg3(seg)
        tissue = {"CSF": 1, "GM": 2, "WM": 3}[tissue]
    mask = seg == tissue
    return img * mask, mask


def load_results_img(layout, id, extra, postprocessed=False, type_suffix=None, skull_stripping=False,
                     seg_source=None, tissue=None):
    """Prediction of ``layout`` for ``extra`` (and the segmentation used to skull-strip it).

    Args:
        type_suffix: image type suffix; defaults to ``"_POST"`` if ``postprocessed`` else ``""``.
        seg_source: ``(extra, type_suffix, postprocessed)`` of the segmentation used for skull
            stripping. Defaults to the prediction's own SEG_96. The GAN viewers used the
            postprocessed segmentation of the selected checkpoint for every checkpoint
            (e.g. ``("28", "_POST", True)`` for MPGAN), since raw predictions have no segmentation.
    """
    if type_suffix is None:
        type_suffix = "_POST" if postprocessed else ""
    img = io.load_nifti(layout.get_result_image_path_name(id, "T1" + type_suffix, (extra,), postprocessed))[0]
    seg = None
    if skull_stripping or seg_source is None:
        seg_extra, seg_suffix, seg_post = seg_source or (extra, type_suffix, postprocessed)
        seg = io.load_nifti(layout.get_result_image_path_name(id, "SEG_96" + seg_suffix, (seg_extra,), seg_post))[0]
    if skull_stripping:
        img = skull_stripping_and_normalize(img, seg)
        img, seg = apply_tissue_mask(img, seg, tissue)
    return img, seg


def _result_names(extras, name_prefix):
    return [f"{name_prefix}{'_'.join(map(str, e)) if isinstance(e, tuple) else e}".upper() for e in extras]


def visualize_results(method, id, extras, postprocessed=False, type_suffix=None, skull_stripping=True, seg_source=None,
                      normalize_if_not_stripped=False, name_prefix="", title=None, **viewer_kwargs):
    """HUNT3, HUNT4 and the predictions for each ``extra``."""
    layout = ResultsLayout(method)
    img_list = [load_original_img(id, 3, skull_stripping, normalize_if_not_stripped)[0],
                load_original_img(id, 4, skull_stripping, normalize_if_not_stripped)[0]]
    for extra in extras:
        img_list.append(load_results_img(layout, id, extra, postprocessed, type_suffix, skull_stripping, seg_source)[0])
    name_list = ["HUNT3", "HUNT4"] + _result_names(extras, name_prefix)
    kwargs = dict(mooving_plane=2, custom_disposition=(2, -1) if len(img_list) > 4 else (1, -1), subtitle_help=(20, 5), subtitle_space=20)
    kwargs.update(viewer_kwargs)
    display.imgshow_3D_list(f"Results {title or method} {id}", img_list, subimg_titles=name_list, **kwargs)


def visualize_results_diff(method, id, extras, postprocessed=False, type_suffix=None, seg_source=None, tissue=None,
                           name_prefix="", title=None, hunt3_first=False, **viewer_kwargs):
    """|image - HUNT4| maps; ``hunt3_first=True`` reproduces the diffusion-model layout (HUNT3 diff, then HUNT4)."""
    layout = ResultsLayout(method)
    img_org_hunt4 = load_original_img(id, 4, skull_stripping=True, tissue=tissue)[0]
    img_org_hunt3 = load_original_img(id, 3, skull_stripping=True, tissue=tissue)[0]
    if hunt3_first:
        img_list = [np.abs(img_org_hunt3 - img_org_hunt4), np.abs(img_org_hunt4 - img_org_hunt4)]
        name_list = ["HUNT3", "HUNT4"]
    else:
        img_list = [np.ones_like(img_org_hunt4), np.abs(img_org_hunt3 - img_org_hunt4)]  # ones: hot colormap spans [0, 1]
        name_list = ["HUNT4", "HUNT3"]
    for extra in extras:
        img_res = load_results_img(layout, id, extra, postprocessed, type_suffix, True, seg_source, tissue)[0]
        img_list.append(np.abs(img_res - img_org_hunt4))
    name_list += _result_names(extras, name_prefix)
    kwargs = dict(mooving_plane=2, custom_disposition=(2, -1) if len(img_list) > 4 else (1, -1), subtitle_help=(20, 5),
                  subtitle_space=20, cmap="hot")
    kwargs.update(viewer_kwargs)
    display.imgshow_3D_list(f"Results {title or method} {id}", img_list, subimg_titles=name_list, **kwargs)


def visualize_results_and_seg(method, id, extras, postprocessed=False, type_suffix=None, tissue=None, name_prefix="",
                              title=None, include_hunt3=False, **viewer_kwargs):
    """Images (top row) and RGB segmentations (bottom row): [HUNT3,] HUNT4 and the predictions."""
    layout = ResultsLayout(method)
    refs = ([3] if include_hunt3 else []) + [4]
    images, segs, names = [], [], []
    for hunt in refs:
        img, seg = load_original_img(id, hunt, skull_stripping=True, tissue=tissue)
        images.append(np.stack([img, img, img], axis=3))
        segs.append(ufs.seg_to_rgbseg(seg) / 255)
        names.append(f"HUNT{hunt}")
    for extra in tqdm(extras):
        img, seg = load_results_img(layout, id, extra, postprocessed, type_suffix, True, None, tissue)
        images.append(np.stack([img, img, img], axis=3))
        segs.append(ufs.seg_to_rgbseg(seg) / 255)
    names += _result_names(extras, name_prefix)
    kwargs = dict(mooving_plane=2, custom_disposition=(2, -1), subtitle_help=(20, 5), subtitle_space=20)
    kwargs.update(viewer_kwargs)
    display.imgshow_3D_list(f"Results {title or method} {id}", images + segs, subimg_titles=names + [""] * len(names), **kwargs)
