"""Intensity normalisation, skull stripping and resampling of 3D volumes."""
import numpy as np
from scipy.ndimage import zoom
from skimage import exposure


def get_basic_metrics(np_array):
    min_val = np.min(np_array)
    max_val = np.max(np_array)
    mean_val = np.mean(np_array)
    std_val = np.std(np_array)
    return min_val, max_val, mean_val, std_val


def get_basic_metrics_str(np_array):
    txt = 'min: {:.4f}, max: {:.4f}, mean: {:.4f}, std: {:.4f}'.format(*get_basic_metrics(np_array))
    return txt


def normalize(img, mask=None, percentile=(0,100)):
    if mask is not None:
        arr_min = np.percentile(img[mask>0], percentile[0])
        arr_max = np.percentile(img[mask>0], percentile[1])
    else:
        arr_min = np.percentile(img, percentile[0])
        arr_max = np.percentile(img, percentile[1])

    img_norm = (img - arr_min) / (arr_max - arr_min)
    img_norm = np.clip(img_norm, 0, 1)
    return img_norm


def standarize(img):
    mean = np.mean(img, axis=(0, 1, 2))  
    std = np.std(img, axis=(0, 1, 2)) 
    return (img - mean) / std


def histogram_matching(img, img_reference):
    return exposure.match_histograms(img, img_reference)


def skull_stripping_and_normalize(img, mask, percentile=(0,100), remove_negatives=True):
    img[mask == 0] = 0
    if remove_negatives:
        img[img < 0] = 0
    img = normalize(img, mask, percentile)
    return img


def resize_3D_image(img, new_shape, binary=False):
    zoom_factors = [n / o for n, o in zip(new_shape, img.shape)]
    if binary:
        resized_mask = zoom(img, zoom_factors, order=0)
        return resized_mask
    return zoom(img, zoom_factors, order=3)
