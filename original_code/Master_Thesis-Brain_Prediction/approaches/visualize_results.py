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

DPI = 1000


def get_images_path_name(id, img_names):
    imgs_path_name = []
    seg_path_name = []
    if "HUNT3" in img_names:
        imgs_path_name.append(pil.get_preprocessed_image_path_name(3, id, absolute=True))
        seg_path_name.append(pil.get_segmentation_path_name(3, id, 96, absolute=True))
    if "HUNT4" in img_names:
        imgs_path_name.append(pil.get_preprocessed_image_path_name(4, id, absolute=True))
        seg_path_name.append(pil.get_segmentation_path_name(4, id, 96, absolute=True))
    if "REG" in img_names:
        imgs_path_name.append(pil.get_elastix_registered_image_path_name(id, type_img="T1", absolute=True))
        seg_path_name.append(pil.get_elastix_registered_image_path_name(id, type_img="SEG_96", absolute=True))
    if "MEAN" in img_names:
        imgs_path_name.append(f"{current_path}/dfield_mean/images/results/{id}/{id}_0_T1_100.nii.gz")
        seg_path_name.append(f"{current_path}/dfield_mean/images/results/{id}/{id}_0_SEG_96_100.nii.gz")
    if "SIM_IMG" in img_names:
        imgs_path_name.append(f"{current_path}/dfield_similar_images/images/results/{id}/{id}_0_T1_DICE_MEAN_100.nii.gz")
        seg_path_name.append(f"{current_path}/dfield_similar_images/images/results/{id}/{id}_0_SEG_96_DICE_MEAN_100.nii.gz")
    if "SIM_IMG_REG" in img_names:
        imgs_path_name.append(f"{current_path}/dfield_similar_images_registered/images/results/{id}/{id}_0_T1_BSP_FAST_5.nii.gz")
        seg_path_name.append(f"{current_path}/dfield_similar_images_registered/images/results/{id}/{id}_0_SEG_96_BSP_FAST_5.nii.gz")
    if "SIM_TIS" in img_names:
        imgs_path_name.append(f"{current_path}/dfield_similar_tissues/images/results/{id}/{id}_0_T1_96_10.nii.gz")
        seg_path_name.append(f"{current_path}/dfield_similar_tissues/images/results/{id}/{id}_0_SEG_96_96_10.nii.gz")
    if "SIM_PATCH" in img_names:
        imgs_path_name.append(f"{current_path}/dfield_similar_patches/images/results/{id}/{id}_0_T1_96_10.nii.gz")
        seg_path_name.append(f"{current_path}/dfield_similar_patches/images/results/{id}/{id}_0_SEG_96_96_10.nii.gz")
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
    return imgs_path_name, seg_path_name


def get_family_images_path_name(id, family):
    if family == "DF":
        img_names = ["HUNT3", "HUNT4", "REG", "SIM_IMG", "SIM_IMG_REG", "SIM_TIS", "SIM_PATCH", "SIM_PATCH_POST"]
    elif family == "GAN":
        img_names = ["HUNT3", "HUNT4", "MPGAN", "MPGAN_SEG", "MGAN", "StyleGAN"]
    elif family == "BEST":
        img_names = ["HUNT3", "HUNT4", "SIM_PATCH_POST", "MPGAN_SEG"]
    elif family == "MNI":
        img_names = ["MNI"]
    elif family == "MEAN":
        img_names = ["HUNT3", "HUNT4", "MEAN"]
    return get_images_path_name(id, img_names), img_names


def visualize_HUNT_SEG_images(ids, hunt=4, nb_tissues=3):
    nb_ids = len(ids)
    img_list = [None] *nb_ids*2
    img_names = [None] *nb_ids*2
    for i, id in enumerate(tqdm(ids)):
        img_hunt4 = nfc.load_nifti(pil.get_preprocessed_image_path_name(hunt, id, absolute=True))[0]
        seg_hunt4 = nfc.load_nifti(pil.get_segmentation_path_name(hunt, id, nb_tissues, absolute=True))[0]
        if nb_tissues > 3:
            img_hunt4 = np.stack([img_hunt4, img_hunt4, img_hunt4], axis=3)
            seg_hunt4 = ufs.seg_to_rgbseg(seg_hunt4)/255
        else:
            seg_hunt4 /= nb_tissues
        img_list[i], img_list[i + nb_ids] = img_hunt4, seg_hunt4
        img_names[i], img_names[i + nb_ids] = f"{id}", f"{id}"
    fc.imgshow_3D_list(f"HUNT4 and SEG {nb_tissues}", img_list, subimg_titles=img_names, subtitle_help=(20,10), custom_disposition=(2,-1), subtitle_space=10, mooving_plane=mooving_plane, current_plane=current_plane)


def visualize_HUNT3_HUNT4_images(ids):
    nb_ids = len(ids)
    img_list = [None] *nb_ids*2
    img_names = [None] *nb_ids*2
    for i, id in enumerate(tqdm(ids)):
        img_hunt3 = nfc.load_nifti(pil.get_preprocessed_image_path_name(3, id, absolute=True))[0]
        img_hunt4 = nfc.load_nifti(pil.get_preprocessed_image_path_name(4, id, absolute=True))[0]
        img_list[i], img_list[i + nb_ids] = img_hunt3, img_hunt4
        img_names[i], img_names[i + nb_ids] = f"{id}", f"{id}"
    fc.imgshow_3D_list("HUNT3 vs HUNT4", img_list, subimg_titles=img_names, subtitle_help=(20,10), custom_disposition=(2,-1), subtitle_space=10, mooving_plane=mooving_plane, current_plane=current_plane)


def visualize_final_reuslts(id, family="DF"):  
    (imgs_path_name, seg_path_name), img_names = get_family_images_path_name(id, family)
    nb_images = len(imgs_path_name)
    img_list = [None] *nb_images
    for i in tqdm(range(nb_images)):
        img = nfc.load_nifti(imgs_path_name[i])[0]
        seg = nfc.load_nifti(seg_path_name[i])[0]
        img = util.skull_stripping_and_normalize(img, seg)
        img_list[i] = img

    save_dict = {"dpi":DPI, "path_name":pil.get_save_image_path_name(f"{id}_res_{family}_{mooving_plane}_{current_plane}")} if save else None
    fc.imgshow_3D_list(f"Results ID: {id}", img_list, subimg_titles=img_names, subtitle_help=(0,0), custom_disposition=(-1, 4), subtitle_space=40, save_dict=save_dict, mooving_plane=mooving_plane, current_plane=current_plane)



def visualize_final_reuslts_dif(id, family="DF"):
    (imgs_path_name, seg_path_name), img_names = get_family_images_path_name(id, family)

    img_hunt4 = nfc.load_nifti(pil.get_preprocessed_image_path_name(4, id, absolute=True))[0]
    seg_hunt4 = nfc.load_nifti(pil.get_segmentation_path_name(4, id, 96, absolute=True))[0]
    img_hunt4 = util.skull_stripping_and_normalize(img_hunt4, seg_hunt4)

    nb_images = len(imgs_path_name) + 1
    img_list = [None] *nb_images

    img_list[0]  = np.abs(img_hunt4 - img_hunt4)
    img_list[0] = np.ones_like(img_hunt4) # just to make the hot colormap work with the range [0,1]
    for i in tqdm(range(nb_images-1)):
        img = nfc.load_nifti(imgs_path_name[i])[0]
        seg = nfc.load_nifti(seg_path_name[i])[0]
        img = util.skull_stripping_and_normalize(img, seg)
        img_list[i+1] = np.abs(img - img_hunt4)

    img_names = ["HUNT4"] + img_names
    save_dict = {"dpi":DPI, "path_name":pil.get_save_image_path_name(f"{id}_dif_{family}_{mooving_plane}_{current_plane}")} if save else None
    fc.imgshow_3D_list(f"Results ID: {id}", img_list, subimg_titles=img_names, subtitle_help=(0,0), custom_disposition=(-1, 4), subtitle_space=40, cmap="hot", save_dict=save_dict, mooving_plane=mooving_plane, current_plane=current_plane)



def visualize_final_reuslts_seg_dif(id, nb_tissues=96, see_tissues=None, family="DF"):
    (_, seg_path_name), img_names = get_family_images_path_name(id, family)

    seg_hunt4 = nfc.load_nifti(pil.get_segmentation_path_name(4, id, 96, absolute=True))[0]
    if see_tissues is not None:
        seg_hunt4[ufs.get_tissue_mask(seg_hunt4, see_tissues) == 0] = 0
    seg_hunt4 = ufs.merge_seg96_to_seg3(seg_hunt4) if nb_tissues == 3 else (ufs.merge_seg96_to_seg36(seg_hunt4) if nb_tissues == 36 else seg_hunt4)

    nb_images = len(seg_path_name) + 1
    img_list = [None] *nb_images
    img_list[0] = np.zeros_like(seg_hunt4) # just to make the hot colormap work with the range [0,1]
    for i in tqdm(range(nb_images-1)):
        seg = nfc.load_nifti(seg_path_name[i])[0]

        if see_tissues is not None:
            seg[ufs.get_tissue_mask(seg, see_tissues) == 0] = 0

        if nb_tissues == 3:
            seg = ufs.merge_seg96_to_seg3(seg)
        elif nb_tissues == 36:
            seg = ufs.merge_seg96_to_seg36(seg)

        img_list[i+1] = 1-(seg == seg_hunt4).astype(int)

    img_names = ["HUNT4"] + img_names
    save_dict = {"dpi":DPI, "path_name":pil.get_save_image_path_name(f"{id}_seg_dif_{family}_{mooving_plane}_{current_plane}")} if save else None
    fc.imgshow_3D_list(f"Results ID: {id}", img_list, subimg_titles=img_names, subtitle_help=(0,0), custom_disposition=(-1, 4), subtitle_space=40, save_dict=save_dict, mooving_plane=mooving_plane, current_plane=current_plane)


def visualize_final_reuslts_seg(id, nb_tissues=96, see_tissues=None, family="DF"):
    (imgs_path_name, seg_path_name), img_names = get_family_images_path_name(id, family)

    nb_images = len(imgs_path_name)
    img_list = [None] *nb_images*2
    for i in tqdm(range(nb_images)):
        img = nfc.load_nifti(imgs_path_name[i])[0]
        seg = nfc.load_nifti(seg_path_name[i])[0]
        img = util.skull_stripping_and_normalize(img, seg)

        if see_tissues is not None:
            seg[ufs.get_tissue_mask(seg, see_tissues) == 0] = 0
        
        if nb_tissues > 3:
            if nb_tissues == 36:
                seg = ufs.merge_seg96_to_seg36(seg)
            img = np.stack([img, img, img], axis=3)
            seg = ufs.seg_to_rgbseg(seg)/255
        else:
            seg = ufs.merge_seg96_to_seg3(seg) / 3     

        img_list[i], img_list[i + nb_images] = img, seg

    img_names = img_names + [""]*nb_images
    save_dict = {"dpi":DPI, "path_name":pil.get_save_image_path_name(f"{id}_seg_{family}_{mooving_plane}_{current_plane}")} if save else None
    fc.imgshow_3D_list(f"Results ID: {id}", img_list, subimg_titles=img_names, subtitle_help=(0,0), custom_disposition=(2,-1), subtitle_space=40, save_dict=save_dict, mooving_plane=mooving_plane, current_plane=current_plane)



def important_tissues2rgb(seg, paires_tissues, colors):
    rgb = np.zeros((*seg.shape, 3), dtype=np.float32)

    c = 0
    for i in range(0, len(paires_tissues), 2):
        rgb[seg == paires_tissues[i]] = colors[c]
        rgb[seg == paires_tissues[i+1]] = colors[c]
        c += 1
    return rgb


def visualize_final_reuslts_seg_important_tissues(id, nb_tissues=96, see_tissues=None, family="DF"):
    (imgs_path_name, seg_path_name), img_names = get_family_images_path_name(id, family)

    nb_images = len(imgs_path_name)
    img_list = [None] *nb_images*2
    for i in tqdm(range(nb_images)):
        img = nfc.load_nifti(imgs_path_name[i])[0]
        seg96 = nfc.load_nifti(seg_path_name[i])[0]
        img = util.skull_stripping_and_normalize(img, seg96)

        seg36 = ufs.merge_seg96_to_seg36(seg96)
        seg3 = ufs.merge_seg96_to_seg3(seg96)

        seg96[ufs.get_tissue_mask(seg96, see_tissues) == 0] = 0
        seg36[ufs.get_tissue_mask(seg36, see_tissues) == 0] = 0

        seg36[seg96>0] = seg96[seg96>0]
        seg3[seg36>0] = 0

        seg_rgb = important_tissues2rgb(seg36, see_tissues, list(colors.values()))/255
        seg3 = np.stack([seg3, seg3, seg3], axis=3)/3
        seg_rgb = seg_rgb + seg3

        img = np.stack([img, img, img], axis=3)
        
        img_list[i], img_list[i + nb_images] = img, seg_rgb

    img_names = img_names + [""]*nb_images
    save_dict = {"dpi":DPI, "path_name":pil.get_save_image_path_name(f"{id}_segI_{family}_{mooving_plane}_{current_plane}")} if save else None
    fc.imgshow_3D_list(f"Results ID: {id}", img_list, subimg_titles=img_names, subtitle_help=(0,0), custom_disposition=(2,-1), subtitle_space=40, save_dict=save_dict, mooving_plane=mooving_plane, current_plane=current_plane)





val_images = np.array(["08202"] + pil.get_matching_ids()[621:])
LOW = 78 #17112 *ax(2) = 96, sag(0) = 77, cor(1) = 122
MEDIUM = 80 #17310 #ax = 96, sag = 75, cor = 122
HIGH = 48 #15170 #ax = 103, sag = 80, cor = 122

save = False
mooving_plane=2
current_plane=110


# visualize_HUNT3_HUNT4_images(val_images[[LOW, MEDIUM, HIGH]])
# visualize_HUNT_SEG_images(val_images[[LOW, MEDIUM, HIGH]], hunt=3, nb_tissues=3)
visualize_final_reuslts(val_images[LOW], family="GAN")
# visualize_final_reuslts_dif(val_images[LOW], family="MEAN")
# visualize_final_reuslts_seg_dif(val_images[HIGH], nb_tissues=36, see_tissues=None, family="BEST")
# visualize_final_reuslts_seg(val_images[LOW], nb_tissues=3, family="DF")
# visualize_final_reuslts_seg_important_tissues(val_images[LOW], nb_tissues=96, see_tissues=IMPORTANT_TISSUES, family="MEAN")


# save = True
# mooving_plane=2
# current_plane=103
# visualize_final_reuslts_dif(val_images[HIGH], family="BEST")
# visualize_final_reuslts_seg_important_tissues(val_images[HIGH], nb_tissues=96, see_tissues=IMPORTANT_TISSUES, family="BEST")


save = True
mooving_plane=1
current_plane=69
# visualize_final_reuslts_seg_important_tissues("MNI", nb_tissues=96, see_tissues=IMPORTANT_TISSUES, family="MNI")
