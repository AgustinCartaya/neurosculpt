import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../'))

sys.path.append(base_path)

import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
from tqdm import tqdm
import pandas as pd
from IPython.display import display


import project_images_loader as pil
import utils.util as util
import utils.functions as fc
import utils.nifti_functions as nfc
import utils.util_freesurfer_segmentation as ufs
import utils.util_dfield_metrics as util_dfield_metrics
import neurite as ne
from scipy import ndimage


IMPORTANT_TISSUES = [42, 3, 43, 4, 53, 17, 49, 10, 2006, 1006, 2009, 1009, 2029, 1029, 2027, 1027, 2028, 1028, 2025, 1025, 2005, 1005]
colors = {
    "red": (255, 0, 0),#cortex (R:42, L:3)
    "purple": (128, 0, 128),#lateral ventricles (R:43, L:4)
    "yellow": (225, 225, 0),#hipocampus (R:53, L:17)
    "blue": (0, 0, 255),#thalamus (R:49, L:10)
    "green": (0, 255, 0),#entorhinal cortex (R:2006, L:1006)
    "cyan": (0, 255, 255),#inferior temporal gyri (R:2009, L:1009)
    "magenta": (255, 0, 255),#superior parietal (R:2029, L:1029)
    "orange": (255, 165, 0),#middle frontal gyrus (R:2027, L:1027)
    "pink": (255, 192, 203),#superior frontal gyrus (R:2028, L:1028)
    # "lime": (0, 255, 127),
    "lime": (10, 128, 90),#precuneus (R:2025, L:1025)
    "brown": (165, 42, 42),#cuneus (R:2005, L:1005)
}



def get_images_path_name(id, img_names, return_df=False):

    imgs_path_name = []
    seg_path_name = []
    df_path_name = []
    if "HUNT3" in img_names:
        imgs_path_name.append(pil.get_preprocessed_image_path_name(3, id, absolute=True))
        seg_path_name.append(pil.get_segmentation_path_name(3, id, 96, absolute=True))
    if "HUNT4" in img_names:
        imgs_path_name.append(pil.get_preprocessed_image_path_name(4, id, absolute=True))
        seg_path_name.append(pil.get_segmentation_path_name(4, id, 96, absolute=True))
    if "REG" in img_names:
        imgs_path_name.append(pil.get_elastix_registered_image_path_name(id, type_img="T1", absolute=True))
        seg_path_name.append(pil.get_elastix_registered_image_path_name(id, type_img="SEG_96", absolute=True))
        df_path_name.append(pil.get_elastix_registered_image_path_name(id, type_img="DFIELD", absolute=True))
    if "SIM_IMG" in img_names:
        df_path_name.append(f"{current_path}/dfield_similar_images/images/dfields/{id}/{id}_IMGSIM_DFIELD_DICE_MEAN_100.nii.gz")
    if "SIM_IMG_REG" in img_names:
        df_path_name.append(f"{current_path}/dfield_similar_images_registered/images/dfields/{id}/{id}_IMGSIM_DFIELD_BSP_FAST_5.nii.gz")
    if "SIM_IMG_REG_REG" in img_names:
        imgs_path_name.append(f"{current_path}/dfield_similar_images_registered/similar_images_registrations/images/17112/17112_{id}_T1_BSP_FAST.nii.gz")
        df_path_name.append(f"{current_path}/dfield_similar_images_registered/similar_images_registrations/images/17112/17112_{id}_DFIELD_BSP_FAST.nii.gz")
        seg_path_name.append(f"{current_path}/dfield_similar_images_registered/similar_images_registrations/images/17112/17112_{id}_SEG_96_BSP_FAST.nii.gz")
    if "SIM_TIS" in img_names:
        df_path_name.append(f"{current_path}/dfield_similar_tissues/images/dfields/{id}/{id}_TISSUESIM_DFIELD_96_100.nii.gz")
    if "SIM_PATCH" in img_names:
        df_path_name.append(f"{current_path}/dfield_similar_patches/images/dfields/{id}/{id}_PATCHSIM_DFIELD_96_10.nii.gz")
    if "SIM_PATCH_POST" in img_names:
        imgs_path_name.append(f"{current_path}/dfield_similar_patches/images/results_postprocessed/{id}/{id}_0_T1_POST_96_10.nii.gz")
        seg_path_name.append(f"{current_path}/dfield_similar_patches/images/results_postprocessed/{id}/{id}_0_SEG_96_POST_96_10.nii.gz")
    if "MPGAN" in img_names:
        imgs_path_name.append(f"{current_path}/MPGAN_paper_16/images/results_postprocessed/{id}/{id}_0_T1_POST_28.nii.gz")
        seg_path_name.append(f"{current_path}/MPGAN_paper_16/images/results_postprocessed/{id}/{id}_0_SEG_96_POST_28.nii.gz")
    if "MPGAN_SEG" in img_names:
        imgs_path_name.append(f"{current_path}/MPGAN_paper_16_seg_loss/images/results_postprocessed/{id}/{id}_0_T1_RAW_52.nii.gz")
        seg_path_name.append(f"{current_path}/MPGAN_paper_16_seg_loss/images/results_postprocessed/{id}/{id}_0_SEG_96_RAW_52.nii.gz")
    if "MGAN" in img_names:
        imgs_path_name.append(f"{current_path}/MGAN_paper_16/images/results_postprocessed/{id}/{id}_0_T1_POST_46.nii.gz")
        seg_path_name.append(f"{current_path}/MGAN_paper_16/images/results_postprocessed/{id}/{id}_0_SEG_96_POST_46.nii.gz")
    if "StyleGAN" in img_names:
        imgs_path_name.append(f"{current_path}/StarGAN_V2_sagital/images/results_postprocessed/{id}/{id}_0_T1_POST_BEST.nii.gz")
        seg_path_name.append(f"{current_path}/StarGAN_V2_sagital/images/results_postprocessed/{id}/{id}_0_SEG_96_POST_BEST.nii.gz")
    if "MNI" in img_names:
        imgs_path_name.append(f"{current_path}/0MNI_Segmentation/images/MNI_T1.nii.gz")
        seg_path_name.append(f"{current_path}/0MNI_Segmentation/images/MNI_SEG_96.nii.gz")

    if img_names[0] in ["SIM_IMG", "SIM_IMG_REG", "SIM_TIS", "SIM_PATCH"]:
        for i in range(len(img_names)):
            imgs_path_name.append(pil.get_preprocessed_image_path_name(3, id, absolute=True))
            seg_path_name.append(pil.get_segmentation_path_name(3, id, 96, absolute=True))
    # elif img_names[0] in ["SIM_IMG_REG_REG"]:
    #     seg_path_name = []
    #     for i in range(len(img_names)):
    #         seg_path_name.append(pil.get_segmentation_path_name(3, id, 96, absolute=True))
    if return_df:
        return imgs_path_name, seg_path_name, df_path_name
    return imgs_path_name, seg_path_name


def get_family_images_path_name(id, family, return_df=False):
    if type(family) is list:
        img_names = family
    if family == "ONLY_DF":
        img_names = ["REG", "SIM_IMG", "SIM_IMG_REG", "SIM_TIS", "SIM_PATCH"]
    if family == "DF":
        img_names = ["HUNT3", "HUNT4", "REG", "SIM_IMG", "SIM_IMG_REG", "SIM_TIS", "SIM_PATCH", "SIM_PATCH_POST"]
    elif family == "GAN":
        img_names = ["HUNT3", "HUNT4", "MPGAN", "MPGAN_SEG", "MGAN", "StyleGAN"]
    elif family == "BEST":
        img_names = ["HUNT3", "HUNT4", "SIM_PATCH_POST", "MPGAN_SEG"]
    elif family == "MNI":
        img_names = ["MNI"]
    return get_images_path_name(id, img_names, return_df=return_df), img_names


def intercalate_zeros(U, step=2):
    ny, nx = U.shape
    U_new = np.zeros((step * ny, step * nx))
    # Copiar los valores originales en posiciones alternas
    U_new[::step, ::step] = U
    
    return U_new

def closest_divisible(A, B):
    return A - (A % B)


def plot_deformation_field_alone(U, V, bg_img, title="", arrow_scale=0.03, headwidth=2, headlength=3, grid_step=1, width=0.002, save_dict=None):
    div_x, div_y = closest_divisible(bg_img.shape[0], grid_step), closest_divisible(bg_img.shape[1], grid_step)
    bg_img = bg_img[:div_x, :div_y]
    U = U[:div_x, :div_y]
    V = V[:div_x, :div_y]

    U = U.T
    V = V.T
    bg_img = bg_img.T

    # Crear una malla regular basada en la forma de U y V
    ny, nx = U.shape
    x = np.linspace(0, nx, nx)
    y = np.linspace(0, ny, ny)
    X, Y = np.meshgrid(x, y)

    # Reducir la resolución del grid
    U = U[::grid_step, ::grid_step]
    V = V[::grid_step, ::grid_step]
    U = intercalate_zeros(U, step=grid_step)
    V = intercalate_zeros(V, step=grid_step)
    
    # Calcular la magnitud del vector de deformación
    magnitude = np.sqrt(U**2 + V**2)
    
    # Normalizar la magnitud para usarla en el tamaño de las flechas
    norm_magnitude = magnitude / np.max(magnitude)
    
    # Filtrar los vectores nulos
    mask = (np.abs(U) > 0.4) | (np.abs(V) > 0.4)
    X = X[mask]
    Y = Y[mask]
    U = U[mask]
    V = V[mask]
    norm_magnitude = norm_magnitude[mask]
    
    # Hacer el fondo negro transparente
    bg_img_rgba = np.zeros((bg_img.shape[0], bg_img.shape[1], 4))
    bg_img_rgba[..., :3] = bg_img[..., np.newaxis]
    bg_img_rgba[..., 3] = np.where(bg_img == 0, 0, 1)  # Transparencia para el negro

    # Crear el plot
    plt.figure(figsize=(10, 10))
    plt.imshow(bg_img_rgba, extent=[0, nx, 0, ny], origin='lower', cmap='gray')
    plt.quiver(X, Y, U, V, norm_magnitude, scale=1/arrow_scale, cmap='turbo', headwidth=headwidth, headlength=headlength, width=width)
    plt.colorbar(label='Magnitud de la deformación (normalizada)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)

    if save_dict is not None:
        plt.savefig(save_dict["path_name"], dpi=save_dict["dpi"], bbox_inches='tight', transparent=True)

    plt.show()


def plot_deformation_field(U, V, bg_img, title="", arrow_scale=0.03, headwidth=2, headlength=3, grid_step=1, width=0.002, save_dict=None):
    div_x, div_y = closest_divisible(bg_img.shape[0], grid_step),closest_divisible(bg_img.shape[1], grid_step)
    bg_img = bg_img[:div_x, :div_y]
    U = U[:div_x, :div_y]
    V = V[:div_x, :div_y]

    U = U.T
    V = V.T
    bg_img = bg_img.T

    # Crear una malla regular basada en la forma de U y V
    ny, nx = U.shape
    x = np.linspace(0, nx, nx)
    y = np.linspace(0, ny, ny)
    X, Y = np.meshgrid(x, y)

    # Reducir la resolución del grid
    U = U[::grid_step, ::grid_step]
    V = V[::grid_step, ::grid_step]
    U = intercalate_zeros(U, step=grid_step)
    V = intercalate_zeros(V, step=grid_step)
    
    # Calcular la magnitud del vector de deformación
    magnitude = np.sqrt(U**2 + V**2)
    
    # Normalizar la magnitud para usarla en el tamaño de las flechas
    norm_magnitude = magnitude / np.max(magnitude)
    
    # Filtrar los vectores nulos
    mask = (np.abs(U) > 0.4) | (np.abs(V) > 0.4)
    X = X[mask]
    Y = Y[mask]
    U = U[mask]
    V = V[mask]
    norm_magnitude = norm_magnitude[mask]    
    
    # Crear el plot
    plt.figure(figsize=(10, 10))
    plt.imshow(bg_img, extent=[0, nx, 0, ny], origin='lower', cmap='gray')
    plt.quiver(X, Y, U, V, norm_magnitude, scale=1/arrow_scale, cmap='turbo', headwidth=headwidth, headlength=headlength, width=width)
    plt.colorbar(label='Magnitud de la deformación (normalizada)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)

    if save_dict is not None:
        plt.savefig(save_dict["path_name"], dpi=save_dict["dpi"], bbox_inches='tight')

    plt.show()


def normalize_vectors(vectors):
    magnitudes = np.linalg.norm(vectors, axis=-1, keepdims=True)
    magnitudes[magnitudes == 0] = 1
    return vectors / magnitudes

def calculate_magnitude(vectors):
    U = vectors[..., 0]
    V = vectors[..., 1]
    W = vectors[..., 2]
    magnitudes = np.sqrt(U**2 + V**2 + W**2)
    
    return magnitudes


def get_circular_se(seg_2):
    radio = seg_2 // 2
    elemento = np.zeros((seg_2, seg_2, seg_2))
    centro = seg_2 // 2
    x, y, z = np.ogrid[:seg_2, :seg_2, :seg_2]
    distancia = np.sqrt((x - centro)**2 + (y - centro)**2 + (z - centro)**2)
    elemento[np.where(distancia <= radio)] = 1

    return elemento

def dilate_tissue(mask, se=7):
    SE_1 = get_circular_se(se)
    return ndimage.binary_dilation(mask, structure=SE_1).astype(mask.dtype)


def get_tissue_borders(seg, se=7):
    mask_wm = np.where(seg == 3, 1, 0)
    mask_wm = dilate_tissue(mask_wm, se=se)

    mask_gm = np.where(seg == 2, 1, 0)
    mask_csf = np.where(seg == 1, 1, 0)

    borders_wm_gm = mask_wm * mask_gm
    borders_wm_csf = mask_wm * mask_csf
    return borders_wm_gm + borders_wm_csf

def giva_layer_and_magnitude(df):
    magnitude = calculate_magnitude(df)
    df_norm = normalize_vectors(df[...,:2])
    df_norm *= magnitude[..., np.newaxis]
    return df_norm


def visualize_final_df(id, nb_tissues=96, see_tissues=None, family="REG", extra_info="", only_df=False):
    (imgs_path_name, seg_path_name, df_path_name), img_names = get_family_images_path_name(id, family, return_df=True)
    nb_images = len(imgs_path_name)
    img_list = [None] *nb_images
    for i in tqdm(range(nb_images)):
        img = nfc.load_nifti(imgs_path_name[i])[0]
        seg = nfc.load_nifti(seg_path_name[i])[0]
        df = nfc.load_nifti(df_path_name[i])[0][...,0,:]
        img = util.skull_stripping_and_normalize(img, seg)
        
        if nb_tissues == 3:
            seg = ufs.merge_seg96_to_seg3(seg)
        elif nb_tissues == 36:
            seg = ufs.merge_seg96_to_seg36(seg)

        if see_tissues is not None:
            seg[ufs.get_tissue_mask(seg, see_tissues) == 0] = 0

        seg_mask = np.where(seg>0, 1, 0)
        img = img * seg_mask

        if see_tissues is None and (extra_info in ["sim_tissue","sim_tissue_df"]):
            mask_borders = get_tissue_borders(seg, se=1)
            img[mask_borders == 1] = 1
            seg_mask[mask_borders == 1] = 0

        for layer in range(3):
            df[...,layer] = df[...,layer] * seg_mask

        img_list[i]= df

        layer = 113

        df = giva_layer_and_magnitude(df)
        U = df[:,:,layer, 0]
        V = df[:,:,layer, 1]
        img_bg = img[:,:,layer]



        save_dict = {"dpi":DPI, "path_name":pil.get_save_image_path_name_poster(f"{id}_img_df_{img_names[i]}_{mooving_plane}_{current_plane}_{extra_info}")} if save else None
        if only_df:
            if see_tissues is None and (extra_info in ["sim_tissue","sim_tissue_df"]):
                img_bg = mask_borders[:,:,layer]
            else:
                img_bg = np.zeros_like(img_bg)
            # plot_deformation_field_alone(U, V, img_bg, title=f"{img_names[i]}", arrow_scale=0.013, headwidth=4, headlength=6, grid_step=3, width=0.0025, save_dict=save_dict)
            plot_deformation_field_alone(U, V, img_bg, title=f"{img_names[i]}", arrow_scale=0.019, headwidth=4, headlength=6, grid_step=5, width=0.004, save_dict=save_dict)

        else:
            # plot_deformation_field(U, V, img_bg, title=f"{img_names[i]}", arrow_scale=0.013, headwidth=4, headlength=6, grid_step=5, width=0.0025, save_dict=save_dict)
            plot_deformation_field(U, V, img_bg, title=f"{img_names[i]}", arrow_scale=0.019, headwidth=4, headlength=6, grid_step=5, width=0.004, save_dict=save_dict)
        

def plot_image_alone(bg_img, title="",save_dict=None):
    bg_img = bg_img
    bg_img = bg_img.T

    plt.imshow(bg_img, origin='lower', cmap='gray')
    plt.title(title)
    if save_dict is not None:
        plt.savefig(save_dict["path_name"], dpi=save_dict["dpi"], bbox_inches='tight')

    plt.show()

def visualize_image_alone(id, family="REG", extra_info=""):
    (imgs_path_name, seg_path_name), img_names = get_family_images_path_name(id, family, return_df=False)
    nb_images = len(imgs_path_name)
    for i in tqdm(range(nb_images)):
        img = nfc.load_nifti(imgs_path_name[i])[0]
        seg = nfc.load_nifti(seg_path_name[i])[0]
        img = util.skull_stripping_and_normalize(img, seg)
        img = img[:,:,current_plane]
        # img = img[current_plane,:,:]
        save_dict = {"dpi":DPI, "path_name":pil.get_save_image_path_name_poster(f"{id}_img_{img_names[i]}_{mooving_plane}_{current_plane}_{extra_info}")} if save else None
        plot_image_alone(img, title=f"{img_names[i]}", save_dict=save_dict)


def plot_tissue_alone(bg_img, title="",save_dict=None):
    bg_img = bg_img
    bg_img = bg_img.T

    plt.imshow(bg_img, origin='lower', cmap='gray')
    plt.title(title)
    if save_dict is not None:
        plt.savefig(save_dict["path_name"], dpi=save_dict["dpi"], bbox_inches='tight')

    plt.show()

def visualize_tissue_alone(id, see_tissues, nb_tissues=3, family="REG", extra_info=""):
    (imgs_path_name, seg_path_name), img_names = get_family_images_path_name(id, family, return_df=False)
    nb_images = len(imgs_path_name)
    for i in tqdm(range(nb_images)):
        seg = nfc.load_nifti(seg_path_name[i])[0]

        if nb_tissues == 3:
            seg = ufs.merge_seg96_to_seg3(seg)
        elif nb_tissues == 36:
            seg = ufs.merge_seg96_to_seg36(seg)

        seg[ufs.get_tissue_mask(seg, see_tissues) == 0] = 0
        seg = seg[:,:,current_plane]
        # img = img[current_plane,:,:]
        save_dict = {"dpi":DPI, "path_name":pil.get_save_image_path_name_poster(f"{id}_seg{see_tissues}_{img_names[i]}_{mooving_plane}_{current_plane}_{extra_info}")} if save else None
        plot_tissue_alone(seg, title=f"{img_names[i]}", save_dict=save_dict)

val_images = np.array(["08202"] + pil.get_matching_ids()[621:])



LOW = 78 #17112 *ax(2) = 96, sag(0) = 77, cor(1) = 122 
MEDIUM = 80 #17310 #ax = 96, sag = 75, cor = 122
HIGH = 48 #15170 #ax = 103, sag = 80, cor = 122

save = True
mooving_plane=2
current_plane=113
DPI = 1000

# Most similar patients to 17112: 13329, 12582, 12742

most_similar_images_to_17112 = [13329, 12582, 12742]

# ------- IMAGES  RESULTS
# visualize_image_alone(val_images[LOW], family=["HUNT3"])
# visualize_image_alone(val_images[LOW], family=["HUNT4"])
# visualize_image_alone(val_images[LOW], family=["SIM_PATCH_POST"])
# visualize_image_alone(val_images[LOW], family=["MPGAN_SEG"])

# visualize_image_alone(val_images[LOW], family=["MPGAN"])
# visualize_image_alone(val_images[LOW], family=["MGAN"])
# visualize_image_alone(val_images[LOW], family=["StyleGAN"])
# visualize_image_alone(most_similar_images_to_17112[0], family=["HUNT4"])


# visualize_image_alone(most_similar_images_to_17112[0], family=["SIM_IMG_REG_REG"])
# visualize_image_alone(most_similar_images_to_17112[1], family=["SIM_IMG_REG_REG"])
# visualize_image_alone(most_similar_images_to_17112[2], family=["SIM_IMG_REG_REG"])

visualize_tissue_alone(val_images[LOW], 1, family=["HUNT3"])

visualize_tissue_alone(most_similar_images_to_17112[0], 1, family=["HUNT3"])
visualize_tissue_alone(most_similar_images_to_17112[1], 1, family=["HUNT3"])
visualize_tissue_alone(most_similar_images_to_17112[2], 1, family=["HUNT3"])

# visualize_final_df(most_similar_images_to_17112[0], nb_tissues=3, see_tissues=None, family=["REG"], only_df=True, extra_info="sim_patches")
# visualize_final_df(most_similar_images_to_17112[1], nb_tissues=3, see_tissues=None, family=["REG"], only_df=True, extra_info="sim_patches")
# visualize_final_df(most_similar_images_to_17112[2], nb_tissues=3, see_tissues=None, family=["REG"], only_df=True, extra_info="sim_patches")

# ------- Similar images
# visualize_final_df(most_similar_images_to_17112[0], nb_tissues=3, see_tissues=None, family=["REG"], extra_info="sim_images")
# visualize_final_df(most_similar_images_to_17112[1], nb_tissues=3, see_tissues=None, family=["REG"], extra_info="sim_images")
# visualize_final_df(most_similar_images_to_17112[2], nb_tissues=3, see_tissues=None, family=["REG"], extra_info="sim_images")

# visualize_final_df(val_images[LOW], nb_tissues=3, see_tissues=None, family=["SIM_IMG"], extra_info="sim_images")
# visualize_final_df(val_images[LOW], nb_tissues=3, see_tissues=None, family=["SIM_IMG"], only_df=True, extra_info="sim_images_df")

# ------- Similar images registered
# visualize_final_df(most_similar_images_to_17112[0], nb_tissues=3, see_tissues=None, family=["SIM_IMG_REG_REG"], extra_info="sim_images_reg")
# visualize_final_df(most_similar_images_to_17112[1], nb_tissues=3, see_tissues=None, family=["SIM_IMG_REG_REG"], extra_info="sim_images_reg")
# visualize_final_df(most_similar_images_to_17112[2], nb_tissues=3, see_tissues=None, family=["SIM_IMG_REG_REG"], extra_info="sim_images_reg")

# visualize_final_df(val_images[LOW], nb_tissues=3, see_tissues=None, family=["SIM_IMG_REG"], extra_info="sim_images_reg")
# visualize_final_df(val_images[LOW], nb_tissues=3, see_tissues=None, family=["SIM_IMG_REG"], only_df=True, extra_info="sim_images_reg_df")

# ------- Similar tissues
# visualize_final_df(most_similar_images_to_17112[0], nb_tissues=3, see_tissues=[1], family=["REG"], extra_info="sim_tissue")
# visualize_final_df(most_similar_images_to_17112[1], nb_tissues=3, see_tissues=[2], family=["REG"], extra_info="sim_tissue")
# visualize_final_df(most_similar_images_to_17112[2], nb_tissues=3, see_tissues=[3], family=["REG"], extra_info="sim_tissue")

# visualize_final_df(val_images[LOW], nb_tissues=3, see_tissues=None, family=["SIM_TIS"], extra_info="sim_tissue")
# visualize_final_df(val_images[LOW], nb_tissues=3, see_tissues=None, family=["SIM_TIS"], only_df=True, extra_info="sim_tissue_df")


# ------- Similar patches
# visualize_final_df(most_similar_images_to_17112[0], nb_tissues=3, see_tissues=None, family=["REG"], extra_info="sim_patches")
# visualize_final_df(most_similar_images_to_17112[1], nb_tissues=3, see_tissues=None, family=["REG"], extra_info="sim_patches")
# visualize_final_df(most_similar_images_to_17112[2], nb_tissues=3, see_tissues=None, family=["REG"], extra_info="sim_patches")

# visualize_final_df(val_images[LOW], nb_tissues=3, see_tissues=None, family=["SIM_PATCH"], extra_info="sim_patches")
# visualize_final_df(val_images[LOW], nb_tissues=3, see_tissues=None, family=["SIM_PATCH"], only_df=True, extra_info="sim_patches_df")
