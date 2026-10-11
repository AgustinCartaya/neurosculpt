"""Small numpy helpers: zero padding, slice extraction, batch down-sampling, histograms."""
import numpy as np
from scipy.ndimage import zoom


def pad_2D_img(img, padding=(256, 256)):
    """Zero-pad a 2D image (optionally with a trailing channel axis) to ``padding``, centred."""
    if img.ndim == 3:
        x, y, _ = img.shape
    else:
        x, y = img.shape

    assert x <= padding[0] and y <= padding[1]

    pad_n = max(0, (padding[0] - x) // 2)
    pad_m = max(0, (padding[1] - y) // 2)

    pad_width = (
        (pad_n, pad_n + (padding[0] - x) % 2),
        (pad_m, pad_m + (padding[1] - y) % 2),
    )
    if img.ndim == 3:
        pad_width = pad_width + ((0, 0),)

    return np.pad(img, pad_width, mode='constant', constant_values=0)


def remove_padding(padded_img, original_shape):
    """Inverse of :func:`pad_2D_img`."""
    if padded_img.ndim == 3:
        x, y, _ = original_shape
    else:
        x, y = original_shape

    pad_n = max(0, (padded_img.shape[0] - x) // 2)
    pad_m = max(0, (padded_img.shape[1] - y) // 2)

    if padded_img.ndim == 3:
        return padded_img[pad_n:pad_n + x, pad_m:pad_m + y, :]
    return padded_img[pad_n:pad_n + x, pad_m:pad_m + y]


def extract_layer(array, layer, axis):
    """Return ``array`` indexed at ``layer`` along ``axis``."""
    if axis < 0 or axis >= array.ndim:
        raise ValueError("Invalid axis for the given array.")
    if layer < 0 or layer >= array.shape[axis]:
        raise ValueError("Invalid layer index for the given axis.")

    slicer = [slice(None)] * array.ndim
    slicer[axis] = layer
    return array[tuple(slicer)]


def reduce_3D_images_batch(img_batch, scaling_factor=2, channels_first=False):
    """Down-sample a batch of 3D images (nearest neighbour) by ``scaling_factor``."""
    if not channels_first:
        img_batch = np.transpose(img_batch, (0, 4, 1, 2, 3))

    # channels are on axis 1 from here on
    new_shape = np.array(img_batch.shape[2:]) // scaling_factor
    img_batch_resized = np.zeros((*img_batch.shape[:2], *new_shape), dtype=img_batch.dtype)
    for i in range(img_batch.shape[0]):
        for c in range(img_batch.shape[1]):
            img_batch_resized[i, c] = zoom(img_batch[i, c], zoom=1 / scaling_factor, order=0)

    if not channels_first:
        img_batch_resized = np.transpose(img_batch_resized, (0, 2, 3, 4, 1))
    return img_batch_resized


def normalize_hist(hist, norm="prob"):
    if norm == "l2":
        den = np.sqrt(np.sum(hist**2))
    elif norm == "l1":
        den = np.sum(np.abs(hist))
    elif norm == "prob":
        # NOTE: kept as in the original code, "prob" returns the histogram unnormalised.
        return hist
    return hist / den


def calc_hist(img, normalize=False, range=None, num_bins=256, norm="prob"):
    hist, bins = np.histogram(img.ravel(), bins=num_bins, range=range)
    if normalize:
        hist = normalize_hist(hist, norm=norm)
    return hist, bins


def add_dicts(dict1, dict2, normalize=False):
    """Element-wise sum (or mean with ``normalize=True``) of two dicts with the same keys."""
    if len(dict1) == 0:
        return dict2
    if len(dict2) == 0:
        return dict1

    res = {}
    for key in dict1.keys():
        total = dict1[key] + dict2[key]
        if normalize:
            total /= 2
        res[key] = total
    return res


def divide_dicts_by_integer(dict1, divisor):
    return {key: value / divisor for key, value in dict1.items()}
