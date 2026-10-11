import numpy as np
from scipy.ndimage import zoom
from scipy.interpolate import RegularGridInterpolator

import utils.functions as fc
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from skimage import exposure
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def get_basic_metrics(np_array):
    min_val = np.min(np_array)
    max_val = np.max(np_array)
    mean_val = np.mean(np_array)
    std_val = np.std(np_array)
    return min_val, max_val, mean_val, std_val

def get_basic_metrics_str(np_array):
    txt = 'min: {:.4f}, max: {:.4f}, mean: {:.4f}, std: {:.4f}'.format(*get_basic_metrics(np_array))
    return txt


# ---- preprocessing
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

# ---- resizing images and landmarks
def resize_3D_image(img, new_shape, binary=False):
    zoom_factors = [n / o for n, o in zip(new_shape, img.shape)]
    if binary:
        resized_mask = zoom(img, zoom_factors, order=0)
        return resized_mask
    return zoom(img, zoom_factors, order=3)


def compute_dice_coefficient(mak1, mask2):
    intersection = np.logical_and(mak1, mask2)
    return 2. * intersection.sum() / (mak1.sum() + mask2.sum())


# def compute_multiple_dice_coefficients(masks1, masks2):
#     seg_indexs = list(set(np.unique(masks1).astype(np.int)) | set(np.unique(masks2).astype(np.int)))
#     seg_indexs.sort()
#     dices = {}
#     for seg_index in seg_indexs:
#         mask1 = masks1 == seg_index
#         mask2 = masks2 == seg_index
#         dices[seg_index] = compute_dice_coefficient(mask1, mask2)
#     return dices


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


    


# ---- image visualization
def draw_landmarks(img3D, layer, landmarks=None, mooving_plane=2, lm_radius=5, mask3D=None, mask_tranparence=100):
    if mooving_plane == 0:
        img = img3D[layer].T
    elif mooving_plane == 1:
        img = img3D[:,layer,:].T
    elif mooving_plane == 2:
        img = img3D[...,layer].T

    if landmarks is None:
        return img

    if mooving_plane == 0: # moving in x
        lm = landmarks[landmarks[:,0] == layer][:,1:].astype(np.uint32)
    elif mooving_plane == 1: # moving in y
        lm = landmarks[landmarks[:,1] == layer][:,[0,2]].astype(np.uint32)
    elif mooving_plane == 2: # moving in z
        lm = landmarks[landmarks[:,2] == layer][:, :2].astype(np.uint32)

    img_wp = fc.draw_points(img, lm, radius=lm_radius, color=(0,255,0))   

    if mask3D is not None:
        if mooving_plane == 0:
            mask = mask3D[layer].T
        elif mooving_plane == 1:
            mask = mask3D[:,layer,:].T
        elif mooving_plane == 2:
            mask = mask3D[...,layer].T
        img_wp[...,0] += (mask.astype(np.uint8) * mask_tranparence)
    return img_wp


def imgshow_3D(name, img3D, landmarks=None, mask=None, current_plane=100, cmap='gray', mooving_plane=2, lm_radius=5, mask_to_draw=None, mask_tranparence=20):
    img3D = (normalize(img3D, mask)*255).astype(np.uint8)
    
    # Crear una figura y ejes para la imagen
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Mostrar la imagen inicial
    img = ax.imshow(draw_landmarks(img3D, current_plane, landmarks, mooving_plane=mooving_plane, lm_radius=lm_radius, mask3D=mask_to_draw, mask_tranparence=mask_tranparence), cmap=cmap)

    # Crear un control deslizante para cambiar de plano
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, f'layer {mooving_plane}', 0, img3D.shape[mooving_plane] - 1, valinit=current_plane, valstep=1)

    # Función para actualizar la imagen cuando se cambia el valor del slider
    def update(val):
        current_plane = int(slider.val)
        img.set_data(draw_landmarks(img3D, current_plane, landmarks, mooving_plane=mooving_plane, lm_radius=lm_radius, mask3D=mask_to_draw, mask_tranparence=mask_tranparence))
        fig.canvas.draw_idle()

    ax.set_title(name)
    slider.on_changed(update)
    plt.show()



def plot_2Dhistograms(title, hist, x_edges, y_edges, axs=None, fig=None, x_label="", y_label="", cmap="gray"):
    if axs is None:
        plt.imshow(hist.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', cmap=cmap)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.colorbar()
    else:
        img = axs.imshow(hist.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', cmap=cmap)
        axs.set_xlabel(x_label)
        axs.set_ylabel(y_label)
        axs.set_title(title)
        # fig.colorbar(img, ax=axs)  


def plot_3d_vector_field(data, spacing=4):
    w, h, d = data.shape[:3]
    
    # Preparar los datos para el trazado
    X, Y, Z, U, V, W = [], [], [], [], [], []
    
    for i in range(0,w,spacing):
        for j in range(0,h,spacing):
            for k in range(0,d,spacing//4):
                X.append(i)
                Y.append(j)
                Z.append(k)
                U.append(data[i, j, k, 0])
                V.append(data[i, j, k, 1])
                W.append(data[i, j, k, 2])

    # Crear una figura para trazar
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Trazar los vectores
    ax.quiver(X, Y, Z, U, V, W, length=5)

    # Establecer límites y etiquetas
    ax.set_xlim([0, w])
    ax.set_ylim([0, h])
    ax.set_zlim([0, d])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Mostrar el gráfico
    plt.show()

