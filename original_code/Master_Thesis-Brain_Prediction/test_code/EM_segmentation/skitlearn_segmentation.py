import numpy as np
from sklearn.mixture import GaussianMixture


INIT_K_MEANS = "kmeans"
INIT_RANDOM = "random_from_data"



# -------- add background and reshape labels
def stack_images(imgs):
    """
    convert images in flat arrays and stack them in a matrix (nxm)
    - n: is the number of images
    - m: are the number of elements of the first image (all images are of the same size)
    """
    d = len(imgs)
    staked_images = np.zeros((d, imgs[0].size), dtype=imgs[0].dtype)
    for z in range(d):
        staked_images[z] = imgs[z].ravel()

    return staked_images

# Remove the background of the flat images
def remove_colums(data, binary_mask):
    mask = binary_mask.ravel() > 0
    data_new = data[:,mask]
    return data_new

# Add background to the labels
def insert_columns(flat_imgs, binary_mask):
    result = np.zeros_like(binary_mask)
    mask = binary_mask > 0
    result[mask] = flat_imgs
    return result

# Reshape a flat image to coincide with the shape of the original image
def reshape_flat_img(flat_img, img_shape):
    if len(img_shape) == 2:
        reshaped_img = flat_img.reshape(img_shape[0], img_shape[1])
    elif len(img_shape) == 3:
        reshaped_img = flat_img.reshape(img_shape[0], img_shape[1], img_shape[2])
    return reshaped_img

# Fix labels for segmentation
def fix_labels(labels, imgs, mask, reshape_labels):
    if mask is not None:
        mask_flat = mask.ravel()
        labels = insert_columns(labels+1, mask_flat)

    if reshape_labels:
        labels = reshape_flat_img(labels, imgs[0].shape)

    return labels


# Sklearn EM
def em_sklearn(imgs, k, initialization=INIT_K_MEANS, min_change=0.005, max_iterations=50, reshape_labels=True, mask=None, plot_dict=None):
    # Create the vector of images
    if mask is not None:
        # Remove the background of the images
        staked_imgs = stack_images(imgs)
        staked_imgs = remove_colums(staked_imgs, mask).T
    else:
        staked_imgs = stack_images(imgs).T
        
    gmm = GaussianMixture(n_components=k, max_iter=max_iterations, init_params=initialization, tol=min_change)
    print(staked_imgs.shape)
    gmm.fit(staked_imgs)
    labels = gmm.predict(staked_imgs)

    # # Print the number of iterations
    # print(gmm.n_iter_)

    # Plot the final results if requested
    if plot_dict is not None:
        plot_dict["function"](staked_imgs.T, gmm.means_.T, gmm.covariances_.transpose((1, 2, 0)), labels, plot_dict["nb_points"])

    labels = fix_labels(labels, imgs, mask, reshape_labels)

    return labels, gmm.means_, gmm.covariances_


def change_labels(img, segmentation, img_gt):
    labels_gt = np.unique(img_gt)
    labels_segmentation = np.unique(segmentation)

    correspondence_dictionary = {}

    for label1 in labels_gt:
        min_difference = None
        label = None
        for label2 in labels_segmentation:
            difference = np.abs(np.mean(img[segmentation==label2]) - np.mean(img[img_gt==label1]))
            if min_difference==None or difference<min_difference:
                min_difference = difference
                label = label2

        correspondence_dictionary[label1] = label
        index = np.where(labels_segmentation == label)
        labels_segmentation = np.delete(labels_segmentation, index)

    labels_fixed = np.zeros(img_gt.shape)
    for original_label, new_label in correspondence_dictionary.items():
        labels_fixed[segmentation==original_label] = new_label

    return labels_fixed