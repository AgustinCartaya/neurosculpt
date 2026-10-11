import itertools

import numpy as np

from scipy.ndimage import spline_filter
from scipy.ndimage import distance_transform_edt


from skimage.util import view_as_windows


def extract_patches(volume, patch_shape, extraction_step):
    patches = view_as_windows(volume, patch_shape, step=extraction_step)
    ndim = len(volume.shape)
    npatches = np.prod(patches.shape[:ndim])
    return patches.reshape((npatches, ) + patch_shape)


def perform_voting(patches, patch_shape, expected_shape, extraction_step, window_sep=(16, 16, 16), spline=False):
    # will contain the weighted sum of all the patches that overlap each voxel (numerator in the weighted average)
    vote_img = np.zeros(expected_shape)
    # will contain the sum of the weights of all the patches that overlap each voxel (denominator in the weighted average)
    vote_count = np.zeros(expected_shape)

    # generate all possible indexes for the patches in the image
    coordinates = generate_indexes(patch_shape, extraction_step, expected_shape)

    # Creates a weight matrix based on the distance from the center of the patch
    # on that way the the voxels in the center of the patch will have more weight of the current patch
    # and will be less affected by far away patches
    W = np.ones(patch_shape)
    W[window_sep[0]:-window_sep[0], window_sep[1]:-window_sep[1], window_sep[2]:-window_sep[2]] = 0

    W_dist = distance_transform_edt(W)
    W_dist = 1-W_dist/W_dist.max()

    # create numerator and denominator for the weighted average of the overlapping patches
    for count, coord in enumerate(coordinates) :
        selection = [slice(coord[i] - patch_shape[i], coord[i]) for i in range(len(coord))]
        a, b, c = selection[0], selection[1], selection[2]
        vote_img[a, b, c] += np.multiply(patches[count], W_dist)
        vote_count[a, b, c] += np.multiply(np.ones(vote_img[a, b, c].shape), W_dist)

    vote_count[vote_count == 0] = 1

    w_average = np.divide(vote_img, vote_count)
    # weighted average
    if spline:
        return spline_filter(w_average)
    return w_average


def perform_voting_half_step(patches, patch_shape, expected_shape, extraction_step, window_sep=(16, 16, 16)) :
    # will contain the weighted sum of all the patches that overlap each voxel (numerator in the weighted average)
    vote_img = np.zeros(expected_shape)
    # will contain the sum of the weights of all the patches that overlap each voxel (denominator in the weighted average)
    vote_count = np.zeros(expected_shape)

    # generate all possible indexes for the patches in the image
    coordinates = generate_indexes(patch_shape, extraction_step, expected_shape)

    # Creates a weight matrix 
    W_dist = np.ones(patch_shape) / 4

    # create numerator and denominator for the weighted average of the overlapping patches
    for count, coord in enumerate(coordinates) :
        selection = [slice(coord[i] - patch_shape[i], coord[i]) for i in range(len(coord))]
        a, b, c = selection[0], selection[1], selection[2]
        # vote_img[a, b, c] += patches[count]
        vote_img[a, b, c] += np.multiply(patches[count], W_dist)
        vote_count[a, b, c] += np.multiply(np.ones(vote_img[a, b, c].shape), W_dist)


    vote_count[vote_count == 0] = 1

    # weighted average
    w_average = np.divide(vote_img, vote_count)
    # the spline filter is used to smooth the final image
    # return spline_filter(w_average)
    # return w_average
    return w_average


def generate_indexes(patch_shape, extraction_step, expected_shape) :
    """Generate all possible indexes for the patches in the image, return a nested for-loop generator that iterates over all the expected_shape"""
    ndims = len(patch_shape)
    poss_shape = [patch_shape[i] + extraction_step[i] * ((expected_shape[i] - patch_shape[i]) // extraction_step[i]) for i in range(ndims)]
    idxs = [range(patch_shape[i], poss_shape[i] + 1, extraction_step[i]) for i in range(ndims)]
    return itertools.product(*idxs)