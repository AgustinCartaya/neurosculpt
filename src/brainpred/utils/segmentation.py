"""FreeSurfer/FastSurfer label utilities (tissue merging, cerebellum removal, colour LUT)."""
from pathlib import Path

import pandas as pd
import numpy as np  


# FreeSurfer labels (36 tissues)
CSF = [4, 5, 14, 15, 24, 31, 43, 44, 63]
GM = [3, 8, 10, 11, 12, 13, 16, 17, 18, 26, 28, 42, 47, 49, 50, 51, 52, 53, 54, 58, 60]
WM = [2, 7, 41, 46, 77]
BG = [0]

CEREBELLUM = [7, 8, 15, 16, 46, 47]

CORTEX_3 = [1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035]
CORTEX_42 = [2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2034, 2035]

def read_freesurfer_color_table(names_lower_case=True):
    file_path = Path(__file__).with_name('FreeSurferColorLUT.txt')
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):  # Ignorar las líneas que empiezan con #
                continue
            line = line.strip()  # Eliminar espacios en blanco al principio y al final
            if line:  # Ignorar las líneas vacías
                parts = line.split()  # Dividir la línea en partes por los espacios en blanco
                # Ignorar las líneas que no tienen el número de columnas esperado
                if len(parts) == 6:
                    data.append(parts)  # Ignorar el número de línea y agregar el resto de los datos

    df = pd.DataFrame(data, columns=['ID', 'NAME', 'R', 'G', 'B', 'A'])
    if names_lower_case:
        df.iloc[:, 1] = df.iloc[:, 1].str.lower()
    return df


def segment_freesurfer_labels(img, seg, labels):
    img_masked = np.zeros_like(img)
    color_table = read_freesurfer_color_table()
    for label in labels:
        if isinstance(label, str):
            label = color_table[color_table['NAME'] == str(label)].ID
            label = int(label.iloc[0])            
        if isinstance(label, int):
            img_masked[seg == label] = img[seg == label]
        else:
            raise ValueError(f'Label {label} is not valid')
    return img_masked


def seg_to_rgbseg(seg, colors_dict=None):
    color_table = read_freesurfer_color_table()
    rgb_seg = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for label in np.unique(seg):
        if colors_dict is not None and label in colors_dict:
            color = colors_dict[label]
        else:
            color = color_table[color_table['ID'] == str(int(label))][['R', 'G', 'B']].values
        if len(color) == 0:
            raise ValueError(f'Label {label} not found in the color table')
        rgb_seg[seg == label] = color
    return rgb_seg


def apply_seg_transparency(img, seg, alpha=0.5):
    if len(seg.shape) != 4:
        seg = seg_to_rgbseg(seg)

    img = (255 * (img/np.max(img))).astype(np.uint8)

    img_masked = seg * alpha
    img_masked = img_masked + (1 - alpha) * img
    return img_masked

def get_tissue_mask(seg, tissues):
    tissue_mask = np.isin(seg.astype(int), tissues)
    return tissue_mask

def get_cerebellum_mask(seg_36):
    cerebellum_mask = np.isin(seg_36.astype(int), CEREBELLUM)
    return cerebellum_mask

def remove_cerebellum(seg, seg_36, deep_copy=True, return_cerebellum_mask=False):
    cerebellum_mask = get_cerebellum_mask(seg_36)
    if deep_copy:
        seg_masked = seg.copy()
    else:
        seg_masked = seg
        
    seg_masked[cerebellum_mask == 1] = 0

    if return_cerebellum_mask:
        return seg_masked, cerebellum_mask
    return seg_masked



def get_4_tissues_prob_mask(brain_mask, tissues_3_seg, weights={"bg":0, "ext":1, "csf":1, "gm":1, "wm":1}, dtype=np.float32):
    complete_mask = brain_mask + tissues_3_seg
    multi_channel_mask = np.stack([np.where(complete_mask==0, 1, 0), # bg
                                    np.where(complete_mask==1, 1, 0), # ext
                                    np.where(complete_mask==2, 1, 0), # csf
                                    np.where(complete_mask==3, 1, 0), # gm
                                    np.where(complete_mask==4, 1, 0)], # wm
                                    axis=-1).transpose(3, 0, 1, 2).astype(dtype)
    multi_channel_mask[0][multi_channel_mask[0]>0]=weights["bg"]
    multi_channel_mask[1][multi_channel_mask[1]>0]=weights["ext"]
    multi_channel_mask[2][multi_channel_mask[2]>0]=weights["csf"]
    multi_channel_mask[3][multi_channel_mask[3]>0]=weights["gm"]
    multi_channel_mask[4][multi_channel_mask[4]>0]=weights["wm"]
    return np.sum(multi_channel_mask, axis=0)



# for dice score using gaussian mixture model
def obtain_gmm_elements_using_seg(img, seg, remove_bg=True):
    tissue_index, num_elem = np.unique(seg, return_counts=True)
    if remove_bg and np.any(tissue_index == 0):
        first_index = np.argmax(tissue_index == 0)
        tissue_index = np.delete(tissue_index, first_index)
        num_elem = np.delete(num_elem, first_index)

    nb_tissues = tissue_index.shape[0]
    classs_weights = num_elem/ np.sum(num_elem)
    means = np.zeros(nb_tissues)
    stds = np.zeros(nb_tissues)

    for i in range(nb_tissues):
        tissue_elemnts = img[seg == tissue_index[i]]
        means[i] = np.mean(tissue_elemnts)
        stds[i] = np.std(tissue_elemnts)

    class_imbalance = num_elem / np.max(num_elem)
    class_imbalance = 1/class_imbalance

    return means, stds, classs_weights, class_imbalance

def obtain_multiple_gmm_elements_using_seg(imgs, segs, dtype=np.float32):
    means_list = []
    stds_list = []
    classs_weights_list = []
    class_imbalance_list = []

    for img, seg in zip(imgs, segs):
        means, stds, classs_weights, class_imbalance = obtain_gmm_elements_using_seg(img, seg)
        means_list.append(means)
        stds_list.append(stds)
        classs_weights_list.append(classs_weights)
        class_imbalance_list.append(class_imbalance)
    
    return (np.array(means_list, dtype=dtype), 
            np.array(stds_list, dtype=dtype), 
            np.array(classs_weights_list, dtype=dtype), 
            np.array(class_imbalance_list, dtype=dtype))
     

def merge_seg96_to_seg36(seg96):
    seg_36 = seg96.copy()
    for cortex in CORTEX_3:
        seg_36[seg_36 == cortex] = 3
    for cortex in CORTEX_42:
        seg_36[seg_36 == cortex] = 42
    return seg_36

def merge_seg36_to_seg3(seg_36):
    seg_3 = []
    for tissues in [BG, CSF, GM, WM]:
        mask = np.isin(seg_36, tissues)
        mask = np.where(mask, 1, 0)
        seg_3.append(mask)

    seg_3 = np.stack(seg_3, axis=3)
    seg_3 = np.argmax(seg_3, axis=3).astype(np.uint8)
    return seg_3

def merge_seg96_to_seg3(seg96):
    return merge_seg36_to_seg3(merge_seg96_to_seg36(seg96))
