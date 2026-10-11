
import numpy as np
from sklearn.mixture import GaussianMixture

import functions.functions as fc
import cv2 as cv

INIT_K_MEANS = "kmeans"
INIT_RANDOM = "random_from_data"



# -------- add background and reshape labels
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


# -------- initializations
# ---- centroids
def _initializa_centorid_random(staked_imgs, k):
    """
    this funtion returns a matrix whith random initializations of k
    centroids =        k1     k2      K3
                dim1    0     13      27
                dim2    0.3   0.8     0.9
                dim3    -42   120     30
    """
    num_dimensions = staked_imgs.shape[0]
    k_positions = np.zeros((num_dimensions, k))

    # Find images range
    dim_ranges = []
    for img in staked_imgs:
        dim_ranges.append((np.min(img), np.max(img)))

    # Initialize each dimension
    for dim in range(num_dimensions):  
        k_positions[dim, :] += np.random.uniform(low=dim_ranges[dim][0], high=dim_ranges[dim][1], size=(k))

    return k_positions


def _initialize_centorid_kmeans(staked_imgs, k, min_change=0.001, max_iterations=50, nb_rep=10):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, max_iterations, min_change)
    _, _, centers = cv.kmeans(staked_imgs.T.astype(np.float32), k, None, criteria, nb_rep, cv.KMEANS_PP_CENTERS)
    return centers.T


# ---- labels prob
# Give the same probability for each cluster
def _initialize_labels_prob_equitative(staked_imgs, centroids):
    k = centroids.shape[1]
    labels_prob = np.ones((k, staked_imgs.shape[1]))
    labels_prob = labels_prob/k
    return labels_prob


# Give the probability for each cluster depending on the distance to the centroid
def _initialize_labels_prob_by_distance(staked_imgs, centroids):
    k = centroids.shape[1]

    # compute distance from all centroids for each point of the multiple images
    distances = np.zeros((k, staked_imgs.shape[1]))
    for i in range(k):
        distances[i,:] = np.sqrt(np.sum((staked_imgs - centroids[:,i][:, np.newaxis]) ** 2, axis=0))

    labels_prob = 1 - (distances / np.sum(distances, axis=0))
    labels_prob = labels_prob / np.sum(labels_prob, axis=0)
    return labels_prob


# ---- convariance
def _initialize_convariance_matrix(staked_imgs, centroids, labels_prob, k):
    d = staked_imgs.shape[0]
    covariances = np.zeros((d, d, k))

    inv_labels_sum = 1/np.sum(labels_prob, axis=1)
    for i in range(k):
        x_u = staked_imgs - centroids[:,i][:, np.newaxis]
        new_cov = np.dot(labels_prob[i, :] * x_u, x_u.T) * inv_labels_sum[i]
        covariances[:,:, i] = new_cov

    return covariances



# ---- initialization types
def _k_means_initialization(staked_imgs, k, min_change=0.001, max_iterations=50, nb_rep=10):
    centroids = _initialize_centorid_kmeans(staked_imgs, k, min_change, max_iterations, nb_rep)
    labels_prob = _initialize_labels_prob_equitative(staked_imgs, centroids)
    covariances = _initialize_convariance_matrix(staked_imgs, centroids, labels_prob, k)
    return labels_prob, covariances, centroids


def _random_initialization(staked_imgs, k):
    centroids = _initializa_centorid_random(staked_imgs, k)
    labels_prob = _initialize_labels_prob_equitative(staked_imgs, centroids)
    covariances = _initialize_convariance_matrix(staked_imgs, centroids, labels_prob, k)
    return labels_prob, covariances, centroids




# -------- EM functions
# Calculate the probability of each point given a cluster
def _prob_x_given_k(staked_imgs, centroids, covariances):
    k = centroids.shape[1]
    d = staked_imgs.shape[0] # dimensions
    common_den = (2 * np.pi ) ** d

    res = np.zeros((k, staked_imgs.shape[1]))
    for i in range(k):
        E = covariances[:,:,i]
        den = np.sqrt(common_den * np.linalg.det(E))
        x_u = staked_imgs - centroids[:,i][:, np.newaxis]
        exp = -(1/2) * np.sum(np.dot(x_u.T, np.linalg.inv(E)) * x_u.T, axis=1)
        num = np.exp(exp)
        res[i, :] = num / den
    return res


# Calculate the probability of each cluster
def _prob_k(labels_prob):
    return np.mean(labels_prob, axis=1)


# Calculate the membership weights
def _obtain_labels_prob(staked_imgs, centroids, covariances, old_labels_prob):
    k = centroids.shape[1]

    res = np.zeros((k, staked_imgs.shape[1]))

    prob_x_given_k = _prob_x_given_k(staked_imgs, centroids, covariances)
    prob_k = _prob_k(old_labels_prob)

    w = prob_x_given_k * prob_k[:, np.newaxis]
    den = np.sum(w, axis=0)

    res = w / den
    return res

# Update centorids and covariance matrix
def _update_em(staked_imgs, labels_prob, old_covariance):
    k = labels_prob.shape[0]
    d = staked_imgs.shape[0]

    centroids = np.zeros((d, k))

    # Computing centroids
    inv_labels_sum = 1/np.sum(labels_prob, axis=1)
    for i in range(k):
        for j in range(d):
            centroids[j,i] = np.sum(labels_prob[i] * staked_imgs[j]) 
    centroids = centroids * inv_labels_sum[np.newaxis, :]

    # Computing covariance matices
    covariances = np.zeros((d, d, k))
    for i in range(k):
        x_u = staked_imgs - centroids[:,i][:, np.newaxis]
        new_cov = np.dot(labels_prob[i, :] * x_u, x_u.T) * inv_labels_sum[i]

        # If the determinant is 0, is not possible to calculate the inverse of the matrix
        # Use the previous matrix
        if np.linalg.det(new_cov) == 0 and old_covariance is not None:
            covariances[:,:, i] =  old_covariance[:,:, i]
        else:
            covariances[:,:, i] = new_cov

    return centroids, covariances


# -------- stop condition
def _log_likelihood(labels_prob):
    prob_k = _prob_k(labels_prob)[:, np.newaxis]
    log_likelihood = np.sum(np.log(np.sum(labels_prob * prob_k, axis=1)))
    return log_likelihood


# -------- EM
# Manual EM 
def em(imgs, k, initialization=INIT_K_MEANS, min_change=0.005, max_iterations=50, min_iterations=10, reshape_labels=True, mask=None, plot_dict=None):
    if min_iterations >= max_iterations:
        min_iterations = 0

    # Create the vector of images
    staked_imgs = fc.stack_images(imgs)
    if mask is not None:
        # Remove the background of the images
        staked_imgs = remove_colums(staked_imgs, mask)
        

    # Initialize 
    if initialization == INIT_K_MEANS:
        labels_prob, covariances, centroids = _k_means_initialization(staked_imgs, k, min_change=0.001, max_iterations=50, nb_rep=10)
    elif initialization == INIT_RANDOM:
        labels_prob, covariances, centroids = _random_initialization(staked_imgs, k)
    else:
        raise Exception("INITIALIZATION " + str(initialization) + " DO NOT EXISTS")   
     
    iteration = 0

    log_likelihood = _log_likelihood(labels_prob)
    old_log_likelihood = log_likelihood * 2

    # Main loop
    while (centroids is None or iteration < min_iterations or
           (np.abs(old_log_likelihood - log_likelihood) > min_change and iteration < max_iterations)):
        # print("iteration:", iteration)

        old_log_likelihood = log_likelihood
        labels_prob = _obtain_labels_prob(staked_imgs, centroids, covariances, labels_prob)
        log_likelihood = _log_likelihood(labels_prob)

        # Update parameters
        centroids, covariances = _update_em(staked_imgs, labels_prob, old_covariance=covariances)

        # Plot the results if requested
        if (plot_dict is not None) and (iteration % plot_dict["each"] == 0):
            # print("plot")
            plot_dict["function"](staked_imgs, centroids, covariances, np.argmax(labels_prob, axis=0), plot_dict["nb_points"])

        iteration +=1

    # # Print the number of iterations
    # print(iteration-1)

    labels = np.argmax(labels_prob, axis=0) 

    # Plot the final results if requested
    if plot_dict is not None:
        plot_dict["function"](staked_imgs, centroids, covariances, labels, plot_dict["nb_points"])
    

    labels = fix_labels(labels, imgs, mask, reshape_labels)

    return labels, centroids, covariances


# Sklearn EM
def em_sklearn(imgs, k, initialization=INIT_K_MEANS, min_change=0.005, max_iterations=50, reshape_labels=True, mask=None, plot_dict=None):
    # Create the vector of images
    if mask is not None:
        # Remove the background of the images
        staked_imgs = fc.stack_images(imgs)
        staked_imgs = remove_colums(staked_imgs, mask).T
    else:
        staked_imgs = fc.stack_images(imgs).T
        
    gmm = GaussianMixture(n_components=k, max_iter=max_iterations, init_params=initialization, tol=min_change)
    gmm.fit(staked_imgs)
    labels = gmm.predict(staked_imgs)

    # # Print the number of iterations
    # print(gmm.n_iter_)

    # Plot the final results if requested
    if plot_dict is not None:
        plot_dict["function"](staked_imgs.T, gmm.means_.T, gmm.covariances_.transpose((1, 2, 0)), labels, plot_dict["nb_points"])

    labels = fix_labels(labels, imgs, mask, reshape_labels)

    return labels, gmm.means_, gmm.covariances_











