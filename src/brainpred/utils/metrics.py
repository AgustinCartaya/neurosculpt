"""Image similarity metrics (Dice, NCC, MI, SSIM, PSNR, RMS)."""
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def compute_dice_coefficient(mak1, mask2):
    intersection = np.logical_and(mak1, mask2)
    return 2. * intersection.sum() / (mak1.sum() + mask2.sum())


def compute_multiple_dice_coefficients(masks1, masks2, return_dict=True, seg_indexs=None):
    if seg_indexs is None:
        seg_indexs = np.unique(np.concatenate([masks1, masks2]).astype(np.uint))
    num_segments = len(seg_indexs)
    dices = np.zeros(num_segments)

    for i, seg_index in enumerate(seg_indexs):
        mask1 = masks1 == seg_index
        mask2 = masks2 == seg_index
        dices[i] = compute_dice_coefficient(mask1, mask2)

    if return_dict:
        dices_dict = dict(zip(seg_indexs, dices))
        return dices_dict
    return dices


def compute_mean_dice(masks1, masks2):
    dices = compute_multiple_dice_coefficients(masks1, masks2, return_dict=False)
    return np.mean(dices)


def compute_joint_histogram(img1, img2, n_bins=100, hist_range=None):
    hist_2d, x_edges, y_edges = np.histogram2d(img1.flatten(), img2.flatten(), bins=(n_bins, n_bins), range=hist_range)
    return hist_2d, x_edges, y_edges


def compute_RMS(img1, img2):
    return np.sqrt(np.sum((img1 - img2)**2)) / img1.size


def compute_normalized_cross_correlation(img1, img2):
    mean_img1 = np.mean(img1)
    mean_img2 = np.mean(img2)
    num = np.sum((img1-mean_img1)*(img2-mean_img2))
    den = np.sqrt(np.sum((img1-mean_img1)**2)*np.sum((img2-mean_img2)**2))
    simmilarity = num/den
    return simmilarity


def compute_mutial_information(img1, img2, n_bins=100, hist_range=None):
    if hist_range is None:
        hist_range = ((img1.min(), img1.max()), (img2.min(), img2.max()))
        
    hist_img1, _ = np.histogram(img1, bins=n_bins, range=hist_range[0])
    hist_img2, _ = np.histogram(img2, bins=n_bins, range=hist_range[0])
    hist_2d, x_edges, y_edges = compute_joint_histogram(img1, img2, n_bins=n_bins, hist_range=hist_range)

    hist_img1 = hist_img1.astype(float) 
    hist_img2 = hist_img2.astype(float)
    hist_2d = hist_2d.astype(float)

    hist_img1 /= hist_img1.sum()
    hist_img2 /= hist_img2.sum()
    hist_2d /= hist_2d.sum()

    Ha = -np.sum(hist_img1 * np.log2(hist_img1 + 1e-10))
    Hb = -np.sum(hist_img2 * np.log2(hist_img2 + 1e-10))
    Hab = -np.sum(hist_2d * np.log2(hist_2d + 1e-10))
    simmilarity = Ha + Hb - Hab
    
    return simmilarity


def compute_ssim(img1, img2):
    return ssim(img1, img2, data_range=(img1.max() - img1.min()))


def compute_psnr(img1, img2):
    return psnr(img1, img2, data_range=(img1.max() - img1.min()))


def get_metrics_dict(img_f, img_m):
    # Root Mean Squared
    combined_bg_mask = np.logical_and(img_f > 0, img_m > 0)
    rms = compute_RMS(img_f[combined_bg_mask], img_m[combined_bg_mask])

    # Root Mean Squared
    ncc = compute_normalized_cross_correlation(img_f[combined_bg_mask], img_m[combined_bg_mask])

    # Mutual Information
    hist_range_max = max(img_f.max(), img_m.max())
    hist_range = ((0, hist_range_max), (0, hist_range_max))
    n_bins = 256
    mi  = compute_mutial_information(img_f, img_m, n_bins=n_bins, hist_range=hist_range)

    # compute ssim  
    ssim = compute_ssim(img_f, img_m)

    # {'rms': 0.0, 'ncc': 1.0, 'mi': 1.5, 'ssim': 1.0} perfect values
    return {'rms': rms, 'ncc': ncc, 'mi': mi, 'ssim': ssim}
