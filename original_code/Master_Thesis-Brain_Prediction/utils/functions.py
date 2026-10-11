import numpy as np

import cv2 as cv
import pathlib
import os
import importlib.util
from skimage import io, img_as_bool, transform
from scipy.ndimage import zoom

import matplotlib.pyplot as plt  
from matplotlib.widgets import Slider

from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from itertools import combinations


import random
import math as math 

# ------------ PATHS 
def load_module_from_path(module_name, path_to_file):
    spec = importlib.util.spec_from_file_location(module_name, path_to_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------ STRING 
def str2int_padded(number, nb_zeros = 2):
    min_not_pad = 10**(nb_zeros-1)
    if number<min_not_pad:
        number = str(number).zfill(nb_zeros)
    else:
        number = str(number)
    return number

def get_str_list(from_value, to_value):
    return [str(i) for i in range(from_value, to_value+1)]


# ------------ OS FOLDERS 
def get_base_folder():
    return str(pathlib.Path(__file__).parent.resolve()).replace(os.sep, "/")

# def create_folders(path):
#     # Utiliza os.path.split() para obtener los segmentos de la ruta
#     path = pathlib.Path(path).as_posix()
#     segments = path.split('/')
#     if segments[0] == "":
#         segments = segments[1:]

#     ruta_actual = ""
#     for segment in segments:
#         ruta_actual = os.path.join(ruta_actual, segment)
#         if not os.path.exists(ruta_actual):
#             os.mkdir(ruta_actual)

#     print("Path created:", path)


def create_folders(path):
    # Crear un objeto Path desde la ruta proporcionada
    path = pathlib.Path(path)
    
    # Convertir el iterador parents a una lista y revertirla
    for parent in list(path.parents)[::-1]:
        if not parent.exists():
            parent.mkdir()
    
    # Asegura que el directorio final en la ruta también se cree
    if not path.exists():
        path.mkdir()

    # print("Path created:", path)

def verify_folder(path, create=False):
    if not os.path.exists(path):
        if create:
            create_folders(path)
            if not os.path.exists(path):
                raise Exception("ERROR CREATING:\n" + path)
            return True
        return False
    return True

def verify_file(path_name):
    if os.path.exists(path_name):
        return True
    return False


# ------------ COLOR CONVERSIONS 
def compute_Lab_chroma_img(img):
    img_lan = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    a = (img_lan[:,:,1]).astype(np.float32)
    b = (img_lan[:,:,2]).astype(np.float32)
    img_chroma = np.sqrt(a**2 + b**2)
    return img_chroma

def compute_image_mean_channels(img, channels=[0,1,2], importance=[1,1,1]):
    nb_channels = len(channels)
    img_mean = np.zeros((img.shape[:2]))

    den = 0
    for i in range(nb_channels):
        img_mean += img[:,:,channels[i]].astype(np.float32) * importance[i]
        den += importance[i]

    return (img_mean/den).astype(np.uint8)


# ------------ DATA TYPE CONVERSIONS 
def img_float2unit8(img, color_image = False, max_intensity=255):
    max_val = np.max(img)
    if max_val > 0:
        min_val = np.min(img)
        img_unit8 = max_intensity * ((img - min_val) / (max_val - min_val))
        img_unit8 = img_unit8.astype(np.uint8)
    else:
        img_unit8 = img.astype(np.uint8)

    if color_image:
        img_unit8 = cv.cvtColor(img_unit8,cv.COLOR_GRAY2RGB)

    return img_unit8

def get_color_copy(img):
    if len(img.shape) < 3:
        img_copy = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
    else:
        img_copy = img.copy()
    return img_copy

def list2string(lst, union="", sort_list=False):
    if sort_list:
        lst_sorted = lst.copy()
        lst_sorted.sort()
        return union.join(str(v) for v in lst_sorted)

    else:
        return union.join(str(v) for v in lst)

# ------------ INTERPOLATIONS
def linear_interpolation(value, initial, final):
    return initial + (final - initial) * value

# ------------ SCALE IMAGES VALUES
def scale_image_values(img, v_max=1, percentil=100):
    # remove background
    mask = img != -1
    img_no_bg = img[mask]

    # range 0 - max_vale
    if percentil != 100:
        max_value = np.percentile(img_no_bg, percentil)
    else:
        max_value = np.max(img_no_bg)
    min_value = np.min(img_no_bg)

    img_standarized = v_max * ((img - min_value) / (max_value - min_value))
    img_standarized[~mask] == -1
    
    if percentil != 100:
        img_standarized[img_standarized >= 1] = 1

    return  img_standarized

# ------------ RESIZE IMAGES
def get_resize_dimensions(img, scale_percent=1):
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    return (width, height)


def resize(img, scale):
    return cv.resize(img, get_resize_dimensions(img, scale))


def horizontal_resize(img, w=512, force_h=None, interpolation=None):
    if img is None:
        return None
    if w is None:
        return img.copy()
    
    height, width = img.shape[0], img.shape[1]
    scale_factor = w / width
    new_width = w
    new_height = int(height * scale_factor)
    if force_h is not None:
        new_height += force_h - new_height

    if interpolation is not None:
        resized_image = cv.resize(img, (new_width, new_height), interpolation=interpolation)
    else:
        resized_image = cv.resize(img, (new_width, new_height))
    return resized_image

# ------------ SHOW IMAGES
def imgshow(name, img, scale=1):
    cv.imshow(name, resize(img, scale))

def imgshow_matplotlib(name, img, scale=1, subplot=None, bgr2rgb=False, cmap='gray'):
    width, height = get_resize_dimensions(img, scale)
    if bgr2rgb:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    if subplot is None:
        plt.figure()
        plt.imshow(img, extent=[0, width, 0, height], cmap=cmap)
        plt.title(name)
        plt.axis('off')
    else:
        subplot.imshow(img, extent=[0, width, 0, height], cmap=cmap)
        subplot.set_title(name)
        subplot.axis('off')

# def imgshow_3D(name, img3D, current_plane=100):
#     # Crear una figura y ejes para la imagen
#     fig, ax = plt.subplots()
#     plt.subplots_adjust(bottom=0.25)

#     # Mostrar la imagen inicial
#     img = ax.imshow(img3D[:, :, current_plane], cmap="gray")

#     # Crear un control deslizante para cambiar de plano
#     ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
#     slider = Slider(ax_slider, 'Plano', 0, img3D.shape[2] - 1, valinit=current_plane, valstep=1)

#     # Función para actualizar la imagen cuando se cambia el valor del slider
#     def update(val):
#         current_plane = int(slider.val)
#         img.set_data(img3D[:, :, current_plane])
#         fig.canvas.draw_idle()

#     ax.set_title(name)
#     slider.on_changed(update)
#     plt.show()


def get_3Dimg_plane(img3D, layer, mooving_plane, transpose=False, flip_ax=-1):
    if mooving_plane == 0:
        img = img3D[layer]
    elif mooving_plane == 1:
        img = img3D[:,layer]
    elif mooving_plane == 2:
        img = img3D[:,:,layer]

    if transpose:
        img = np.transpose(img, (1, 0) + tuple(range(2, img.ndim)))
    if flip_ax >=0:
        img = np.flip(img, axis=flip_ax)
    return img

global sliders
sliders = []

def imgshow_3D(name, img3D, current_plane=100, cmap='gray', mooving_plane=2, transpose=True, flip_ax=0, show=True, range_values=[None, None], figsize=(8, 6)):
    """Can plot 3D RGB images"""
    # Crear una figura y ejes para la imagen
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.25)

    # Mostrar la imagen inicial
    img = ax.imshow(get_3Dimg_plane(img3D, current_plane, mooving_plane, transpose=transpose, flip_ax=flip_ax), cmap=cmap, vmin=range_values[0], vmax=range_values[1])

    # Crear un control deslizante para cambiar de plano
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, f'layer {mooving_plane}', 0, img3D.shape[mooving_plane] - 1, valinit=current_plane, valstep=1)

    sliders.append(slider)
    
    # Función para actualizar la imagen cuando se cambia el valor del slider
    def update(val):
        current_plane = int(slider.val)
        img.set_data(get_3Dimg_plane(img3D, current_plane, mooving_plane, transpose=transpose, flip_ax=flip_ax))
        fig.canvas.draw_idle()

    ax.set_title(name)
    slider.on_changed(update)
    if show:
        plt.show()




# show a group of images in the same window
def show_stacked_imgs(imgs, size=6, scale=1, padding_color=[0,0,0]):
    if size%2 != 0:
        size +=1

    # make all images of the same size and convert to rgb if not rgb
    max_w = 0
    max_h = 0
    for i in range(len(imgs)):
        if imgs[i].shape[0] > max_w:
            max_w =  imgs[i].shape[0]
        if  imgs[i].shape[1] > max_h:
            max_h =  imgs[i].shape[1]
        if len(imgs[i].shape) < 3:
            imgs[i] = cv.cvtColor(imgs[i], cv.COLOR_GRAY2BGR)

    for i in range(len(imgs)):
        if imgs[i].shape[0] < max_w or imgs[i].shape[1] < max_h:
            imgs[i] = cv.copyMakeBorder(imgs[i], 0, max_w-imgs[i].shape[0], 0, max_h-imgs[i].shape[1], cv.BORDER_CONSTANT, value=padding_color)

    # adding black images to fill the remaining spaces
    if len(imgs) % size != 0:
        remaining = size - (len(imgs) - (len(imgs) // size)*size) 
        if len(imgs[0].shape) < 3:
            img_black = np.zeros((imgs[0].shape[0],imgs[0].shape[1]), np.uint8)
        else:
            img_black = np.zeros((imgs[0].shape[0],imgs[0].shape[1], 3), np.uint8)

        for i in range(remaining):
            imgs.append(img_black)

    # show imgs
    for i in range(1, len(imgs)+1, size):
        stack_up = np.hstack((imgs[ i-1: int(i+(size/2-1)) ]))
        stack_down = np.hstack((imgs[ int(i+(size/2-1)): int(i+(size-1)) ]))
        stack = np.vstack((stack_up, stack_down))
        imgshow("Res"+str(i) , stack, scale=scale)


# ------------ CONTOURS
def get_contours(img, order_by_area=False, filter_area=None, method = cv.CHAIN_APPROX_SIMPLE):
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, method)

    if order_by_area:
        contours = sorted(contours, key=cv.contourArea, reverse=True)

    if filter_area is not None:
        if filter_area[0] is not None:
            contours = [contour for contour in contours if filter_area[0] < cv.contourArea(contour)]
        if filter_area[1] is not None:
            contours = [contour for contour in contours if cv.contourArea(contour) < filter_area[1] ]
      
    return contours

def draw_contours(img, contours, thickness=-1, color=(255, 255, 255), contour_id=-1):
    img_with_contours = get_color_copy(img)
    cv.drawContours(img_with_contours, contours, contour_id, color, thickness)
    return img_with_contours

def resize_contours(contours, sf):
    for i in range(len(contours)):
        for j in range(len(contours[i])):
            contours[i][j][0][0] = sf * contours[i][j][0][0]
            contours[i][j][0][1] = sf * contours[i][j][0][1]

def get_contours_rois(contours, padding_th=None, img_shape=None):
    rois = []
    if padding_th is not None and img_shape is not None:
        for i in range(len(contours)):
            x, y, w, h = cv.boundingRect(contours[i])

            pad_x = 0
            pad_y = 0
            area = cv.contourArea(contours[i])

            for th, multiplier in padding_th:
                if area < th:
                    pad_x = w * multiplier
                    pad_y = h * multiplier
                    break
            rois.append((max(0, int(x-pad_x)), max(0, int(y-pad_y)), min(img_shape[1], int(x+w+pad_x)), min(img_shape[0], int(y+h+pad_y))))
    else:
        for i in range(len(contours)):
            x, y, w, h = cv.boundingRect(contours[i])
            rois.append((x, y, x+w, y+h))
    return rois

def filter_contour_area(contours, min_area, max_area=-1):
    candidates_filtered = []

    if max_area == -1:
        for contour in contours:
            area = cv.contourArea(contour)
            if area > min_area:
                candidates_filtered.append(contour) 
    else:
        for contour in contours:
            area = cv.contourArea(contour)
            if area > min_area and area < max_area:
                candidates_filtered.append(contour) 

    return candidates_filtered            


def filter_contour_circularity(contours, min_circularity=0, max_circularity=np.inf):
    filtered_contours = []

    for contour in contours:
        area = cv.contourArea(contour)
        longitud = cv.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (longitud * longitud)

        if min_circularity <= circularity <= max_circularity:
            filtered_contours.append(contour)
    return filtered_contours


def draw_contours_with_random_colors(contours, img_size):
    img_res = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    for contour_index in range(len(contours)):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        cv.drawContours(img_res, contours, contour_index, (b,g,r), -1)
    return img_res


def get_contour_centroid(contour):
    M = cv.moments(contour)
    # print(M['m00'])
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return (cx, cy)

def get_contours_centroids(contours):
    centroids = []
    for contour in contours:
        centroids.append(get_contour_centroid(contour))
    return centroids


# ------------ MASKS
def draw_mask(img, mask, thickness=-1, mask_color=(0,255,0)):
    contours,_ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return draw_contours(img, contours, thickness=thickness, color=mask_color, contour_id=-1)

def apply_mask(img, mask, mask_value=0):
    img_masked = img.copy()
    if mask_value == -1:
        img_masked[mask==0] = 0
    else:
        img_masked[mask==255] = mask_value
    return img_masked

def add_patch(img, patch, pos, circular=False):
    if circular:
        pos = [pos[0][0]-pos[1], pos[0][1]-pos[1], 2*pos[1], 2*pos[1]]
    img_restored = img.copy()
    img_restored[pos[1]:pos[1]+pos[3], pos[0]:pos[0]+pos[2]] = patch
    return img_restored   

def create_circle_mask(circle, img_dimensions):
    mask = np.zeros((img_dimensions[0],img_dimensions[1]), np.uint8)
    cv.circle(mask, circle[0], circle[1], (255,255,255), -1)
    return mask

def create_contours_mask(contours, img_dimensions, id=-1):
    mask = np.zeros((img_dimensions[0],img_dimensions[1]), np.uint8)
    cv.drawContours(mask, contours,  id, (255,255,255), -1)
    return mask

def get_values_in_mask(img, mask):
    if len(img.shape) < 3:
        return img[mask > 0]
    else:
        bool_mask = mask > 0
        return img[:,:,0][bool_mask], img[:,:,1][bool_mask], img[:,:,2][bool_mask]


# ------------ ROIS
def draw_roi(img, roi, color=(0, 0, 255), thickness=-1):
    img_copy = get_color_copy(img)
    cv.rectangle(img_copy,
                (roi[0], roi[1]),
                (roi[2], roi[3]),
                color, thickness)
    return img_copy

def draw_rois(img, rois, color=(0, 0, 255), thickness=-1):
    for roi in rois:
        img_copy = draw_roi(img, roi=roi, color=color, thickness=thickness)
    return img_copy

def get_roi(img, roi):
    return img[roi[1]:roi[3], roi[0]:roi[2]]


def create_roi_coords(center, window_size, img_shape):
    if window_size[0] % 2 == 0 or window_size[1] % 2 == 0:
        print("create_roi_coords: window_size SOULD BE ODD WINDOW")
        return None
    
    wroi  = (max(0,center[0] - int(window_size[1]/2)), 
            max(0,center[1] - int(window_size[0]/2)),
            min(img_shape[1], center[0] + int(window_size[1]/2)+1),
            min(img_shape[0], center[1] + int(window_size[0]/2)+1))
    
    return wroi

def get_contained_circle_in_roi(roi, reduction=0):
    roi_w, roi_h = roi[2] - roi[0], roi[3] - roi[1]
    center = (roi[0] + roi_w)//2, (roi[1] + roi_h)//2
    r = int((min(roi_w, roi_h) // 2) * (1-reduction))
    return center, r

def get_img_centered_circle(img_shape, percent=1):
    center = img_shape[1] // 2, img_shape[0] // 2
    r = int((min(img_shape[0], img_shape[1]) // 2) * percent)
    return center, r

# ------------ BORDERS
def get_gradient_img(img):
    gradientX = cv.Sobel(img, cv.CV_64F, 1, 0)
    gradientY = cv.Sobel(img, cv.CV_64F, 0, 1)
    gradient = np.sqrt(np.square(gradientX) + np.square(gradientY))
    return cv.normalize(gradient, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U) # type: ignore

def compute_laplacian_of_gaussian(img):
    img_gaussian = cv.GaussianBlur(img, (3, 3), 0)
    laplaciano_gaussiano = cv.Laplacian(img_gaussian, cv.CV_64F)
    laplaciano_gaussiano = np.uint8(np.absolute(laplaciano_gaussiano))
    return laplaciano_gaussiano

def sharp_edges(img):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv.filter2D(img, -1, kernel)


# ------------ CONTRAST ENHANCEMENT
def gamma_correction(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv.LUT(src, table)


# ------------ GEOMETRIC FIGURES
def resize_circle(circle, sf, translate=False):
    if translate:
        return ((int(circle[0][0]*sf), int(circle[0][1]*sf)),  int(circle[1]*sf))
    else:
        return ((circle[0][0], circle[0][1]),  int(circle[1]*sf))

def resize_rectangle(rect, sf):
    return (int(rect[0]*sf), int(rect[1]*sf), int(rect[2]*sf), int(rect[3]*sf))

def resize_rectangles(rects, sf):
    rects_resized = []
    for rect in rects:
        rects_resized.append(resize_rectangle(rect, sf))
    return rects_resized

def draw_circle(img, circle, color=(0,0,255), thickness=-1):
    img_copy = get_color_copy(img)
    return cv.circle(img_copy, circle[0], circle[1], color, thickness)

def draw_point(img, point, radius=2, color=(0,0,255), thickness=-1):
    img_copy = get_color_copy(img)
    return draw_circle(img_copy, (point, radius), color, thickness)

def draw_points(img, points, radius=2, color=(0,0,255), thickness=-1):
    img_copy = get_color_copy(img)
    for point in points:
        img_copy = draw_point(img_copy, point, radius=radius, color=color, thickness=thickness)
    return img_copy

def draw_line(img, coords, color=(255,255,255), thickness=1):
    img_copy = get_color_copy(img)
    img_copy = cv.line(img_copy, (coords[0], coords[1]), (coords[2], coords[3]), color, thickness)
    return img_copy

def draw_hline(img, y_coord, color=(255,255,255), thickness=1):
    return draw_line(img, (0, y_coord, img.shape[1], y_coord), color=color, thickness=thickness)

def draw_vline(img, x_coord, color=(255,255,255), thickness=1):
    return draw_line(img, (x_coord, 0, img.shape[0], x_coord), color=color, thickness=thickness)

def get_contained_circle(img_shape, percent=1):
    center = img_shape[1] // 2, img_shape[0] // 2
    r = int((min(img_shape[0], img_shape[1]) // 2) * percent)
    return center, r

def get_circle_points(img_size, nb_points, radius, center=None):
    if center is None:
        center = (img_size[1] // 2, img_size[0] // 2)

    angles = np.linspace(0, 2 * np.pi, nb_points, endpoint=False)
    x = radius * np.cos(angles) + center[0] 
    y = radius * np.sin(angles) + center[1]

    x = np.clip(x, 0, img_size[1] - 1)
    y = np.clip(y, 0, img_size[0] - 1)
    
    return np.vstack([x,y]).T.astype(np.int32)

# ------------ TRACE BARS
def create_trackbars(trackbars, window_name="track_bars"):
    cv.namedWindow(window_name,cv.WINDOW_NORMAL)
    for trb in trackbars:
        cv.createTrackbar(trb[0], window_name, trb[1], trb[2], trb[3])

def get_trackbars_values(trackbars_names, window_name):
    values = {}
    for name in trackbars_names:
        values[name] = cv.getTrackbarPos(name, window_name)
    return values


# ------------ METRICS
def calculate_dice(mask, gt):
    cv.normalize(mask, mask, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    cv.normalize(gt, gt, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    ground_truth = img_as_bool(gt)
    mask = img_as_bool(mask)
    intersection = np.logical_and(ground_truth, mask)
    dice = (2.0 * intersection.sum()) / (ground_truth.sum() + mask.sum())
    return dice


# ------------ LISTS
def sort_list(original_list, index):
    sorted_list = original_list.copy()
    sorted_list.sort(key=lambda x:x[index]) 
    interested_value_list = [i[index] for i in sorted_list]
    mean = sum(interested_value_list)/len(interested_value_list)
    median = interested_value_list[int(len(interested_value_list)/2)]
    return sorted_list, mean, median



# ------------ MORPHOLOGIAL OPERATIOS
def break_big_elements(img, img_number, size_th =  400, se=5):
    candidates, _ = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    small_elements = []

    for i in range(len(candidates)):
        area = cv.contourArea(candidates[i])
        if area < size_th:
            small_elements.append(candidates[i])

    SE = cv.getStructuringElement(cv.MORPH_ELLIPSE, (se,se))
    img_open = cv.morphologyEx(img, cv.MORPH_OPEN, SE)

    cv.drawContours(img_open, small_elements, -1, (255, 255, 255), -1)
    return img_open


# ------------ NOTEBOOK
def notebook_show(img, description, separator=" ", show=False, explanation_depth=0, step_depth=0):
    if show and explanation_depth >= step_depth:
        str_description = ""

        if type(description) == str:
            str_description = description

        elif type(description) == tuple or type(description) == list :
            for i in range(len(description)):

                dec_piece = description[i]

                if type(description[i]) != str:
                    dec_piece = str(description[i])

                if i < len(description) -1:
                    str_description += dec_piece + separator
                else:
                    str_description += dec_piece

        print(str_description)
        imagen = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        plt.imshow(imagen)
        plt.axis('off')
        plt.show()



# ------------ HISTOGRAMS
def normalize_hist(hist, norm="prob"):
    if norm == "l2":
        den = np.sqrt(np.sum(hist**2))
    elif norm == "l1":
        den = np.sum(np.abs(hist))
    elif norm == "prob":
        den = np.sum(hist)
        return hist
    return hist / den

def calc_hist(img, normalize=False, range=None, num_bins=256, norm="prob"):  
    hist, bins = np.histogram(img.ravel(), bins=num_bins, range=range)
    if normalize:
        hist = normalize_hist(hist, norm=norm)
    return hist, bins

def plot_hist(img, bins, title="Histogram", range=None):
    # return
    plt.figure(figsize=(8, 6)) 
    plt.hist(img.ravel(), bins=bins, color='blue', alpha=0.7, range=range)  # Histograma
    plt.xlabel('Value')  
    plt.ylabel('Frequency')  
    plt.title(title)  

def get_roi_hist(img, roi_coords, range=None, num_bins=None, normalize=False, ret_bins=False):
    roi = img[roi_coords[1]:roi_coords[3], roi_coords[0]:roi_coords[2]]
    healthy_wroi_hist, healthy_wroi_bins = calc_hist(roi, normalize=normalize, range=range, num_bins=num_bins)
    if ret_bins:
        return healthy_wroi_hist, healthy_wroi_bins
    return healthy_wroi_hist

def plot_precomputed_hist(hist, bins, title="Histogram", color="blue", alpha=0.5, ax=None):
    if ax is None:
        plt.figure(figsize=(8, 6)) 
        plt.bar(bins[:-1], hist, width=(bins[1] - bins[0]), align='edge', alpha=alpha, color=color)
        plt.title(title)
        plt.tight_layout()
    else:
        ax.bar(bins[:-1], hist, width=(bins[1] - bins[0]), align='edge', alpha=alpha, color=color)
        ax.set_title(title)
        # ax.tight_layout()

    


# ------------ COLOR MAPS
def calculate_custom_color_map_value(value, intervals, intervals_colors, color_bk=(0,0,0), color_out_layer=(255,255,255)):
    # background
    if value == -1:
        return color_bk
    
    for i in range(len(intervals)-1):
        if intervals[i] <= value <= intervals[i+1]:
            ratio = (value - intervals[i]) / intervals[i+1]
            blue = linear_interpolation(ratio, intervals_colors[i][0], intervals_colors[i+1][0])
            green = linear_interpolation(ratio, intervals_colors[i][1], intervals_colors[i+1][1])
            red = linear_interpolation(ratio, intervals_colors[i][2], intervals_colors[i+1][2])
            return (blue, green, red)
    
    return color_out_layer
    
def get_img_custom_color_map(img, intervals , intervals_colors, color_bk=(0,0,0), color_out_layer=(255,255,255)):
    """
    converts an angle image to a color image, 
    where the angles that go from 0 (healthy area) to 45 (uncertainty area) go from red to blue 
    and the angles that go from 45 to 90 (tumor area) go from blue to yellow
    colors are in BGR
    the background (-1) is colored black
    """
    img_color_map = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for pix_y in range(img.shape[0]):
        for pix_x in range(img.shape[1]):
            img_color_map[pix_y, pix_x, :] = calculate_custom_color_map_value(img[pix_y, pix_x], 
                                                                     intervals=intervals, 
                                                                     intervals_colors=intervals_colors,
                                                                     color_bk = color_bk,
                                                                     color_out_layer = color_out_layer)
    return img_color_map

def get_img3D_custom_color_map(img, intervals , intervals_colors, color_bk=(0,0,0), color_out_layer=(255,255,255)):

    if intervals == None:
        max_value = np.max(img)
        min_mask = img >= 0
        min_value = np.min(img[min_mask])

        intervals = []
        gap = (max_value - min_value)/(len(intervals_colors)-1)
        for i in range(len(intervals_colors)):
            intervals.append( i * gap +  min_value)

    img_color_map = np.zeros((img.shape[0], img.shape[1], img.shape[1], 3), np.uint8)
    for pix_z in range(img.shape[2]):
        for pix_y in range(img.shape[0]):
            for pix_x in range(img.shape[1]):
                img_color_map[pix_y, pix_x, pix_z, :] = calculate_custom_color_map_value(img[pix_y, pix_x, pix_z], 
                                                                        intervals=intervals, 
                                                                        intervals_colors=intervals_colors,
                                                                        color_bk = color_bk,
                                                                        color_out_layer = color_out_layer)
    return img_color_map



def add_dicts(dict1, dict2, normalize=False):
    if len(dict1)==0:
        return dict2
    if len(dict2)==0:
        return dict1 
    
    res = {}
    for clave in dict1.keys():
        suma = dict1[clave] + dict2[clave]

        if normalize:
            suma /=2

        res[clave] = suma
    return res

def divide_dicts_by_integer(dict1, divisor):
    """
    Divide the values of two dictionaries by an integer divisor.
    """
    result = {}
    for key in dict1.keys():
        result[key] = dict1[key] / divisor

    return result
    
# ------------ STRUCTURAL ELEMENTS
def se_bar(side_enclosing_square_in_px, orientation_in_degrees):
    se_sz = side_enclosing_square_in_px
    sz_ct = side_enclosing_square_in_px // 2
    m = -np.tan(np.radians(orientation_in_degrees))
    [coord_x, coord_y] = np.meshgrid(range(-sz_ct, se_sz-sz_ct), range(-sz_ct, se_sz-sz_ct))

    if m > 1e15:
        distance_to_line = np.abs(coord_x)
    else:
        distance_to_line = np.abs(m * coord_x - coord_y) / np.sqrt(m ** 2 + 1)

    variance = max(1/2, se_sz/14)
    structuring_element = np.exp(-distance_to_line**2 / (2*variance))
    return structuring_element


# ------------ SEGMENTATION ALGORITHMS
def stack_images(imgs):
    """
    convert images in flat arrays and stack them in a matrix (nxm)
    - n: is the number of images
    - m: are the number of elements of the first image (all images are of the same size)
    """
    d = len(imgs)
    staked_images = np.zeros((d, imgs[0].size), dtype=np.uint8)
    for z in range(d):
        staked_images[z] = imgs[z].ravel()

    return staked_images


def expectation_maximization(imgs, k, min_change=0.005, max_iterations=50, reshape_labels=True):
    staked_imgs = stack_images(imgs).T

    # gmm = GaussianMixture(n_components=k, random_state=0, max_iter=max_iterations, init_params="k-means++", tol=min_change)
    gmm = GaussianMixture(n_components=k, random_state=0, max_iter=max_iterations, init_params="kmeans", tol=min_change)
    gmm.fit(staked_imgs)
    labels = gmm.predict(staked_imgs)

    if reshape_labels:
        img_shape = imgs[0].shape
        if len(img_shape) == 2:
            labels = labels.reshape(img_shape[0], img_shape[1])
        elif len(img_shape) == 3:
            labels = labels.reshape(img_shape[0], img_shape[1], img_shape[2])

    return labels


# ------------ PROBABILITIES 
def get_gaussian_image(center, eigen_vector1, eigen_vector2, eigen_value1, eigen_value2, img_shape):
    U = np.array([[eigen_vector1[0], eigen_vector2[0]], 
                  [eigen_vector1[1], eigen_vector2[1]]])
    V = np.diag([eigen_value1, eigen_value2])
    E = np.dot(np.dot(U,V), U.T)

    x, y = np.meshgrid(np.linspace(0, img_shape[1], img_shape[1]), np.linspace(0, img_shape[0], img_shape[0]))

    pos = np.dstack((x, y))
    gaussiana = multivariate_normal(center, E)
    z = gaussiana.pdf(pos)

    z = z/np.max(z)

    return z



# ------------ COMBINATORY
def get_combination_list(n):
    combinations_list = []
    for r in range(1, n + 1):
        combinations_list.extend(list(combinations(range(n), r)))
    return combinations_list



# ------------ MORPHOLOGICAL OPERATIONS
def fast_morph(img, morph="erode", kernel_size=5,  w_size=256):
    img_resized = horizontal_resize(img, w=256, interpolation=cv.INTER_NEAREST)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if morph == "erode":
        img_morph = cv.erode(img_resized, kernel, iterations=1)
    elif morph == "dilate":
        img_morph = cv.dilate(img_resized, kernel, iterations=1)

    img_morph = horizontal_resize(img_morph, w=img.shape[1], force_h=img.shape[0], interpolation=cv.INTER_NEAREST)

    return img_morph


# ------------ DEEP learning
def plot_history(hist):
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    train_accuracy = hist.history['accuracy']
    val_accuracy = hist.history['val_accuracy']

    epochs = range(1, len(train_loss) + 1)

    fig, axs = plt.subplots(1,2, figsize=(12, 8))

    # Plot loss
    axs[0].plot(epochs, train_loss, 'b-', label='Training Loss')
    axs[0].plot(epochs, val_loss, 'r-', label='Validation Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot accuracy
    axs[1].plot(epochs, train_accuracy, 'b-', label='Training acc')
    axs[1].plot(epochs, val_accuracy, 'r-', label='Validation acc')
    axs[1].set_title('Training and Validation acc')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('acc')
    axs[1].legend()

    plt.show()


# ------- DATA AUGMENTATION
def distribute_units(total_units, percentages):
    # Initial allocation with floor rounding using NumPy
    allocation = np.floor(total_units * percentages).astype(int)

    # Calculate remaining units
    units_remaining = total_units - np.sum(allocation)

    # Distribute remaining units
    while units_remaining > 0:
        # Calculate the index of the percentage closest to the next whole number
        idx_max = np.argmax((total_units * percentages) - allocation)
        allocation[idx_max] += 1
        units_remaining -= 1
        
    return allocation

def standardize_and_normalize_image(image):
    # Standardize the image (Z-score normalization)
    image_standardized = (image - np.mean(image)) / np.std(image)

    # Normalize the image to the range 0-1 (Min-Max normalization)
    min_val = np.min(image_standardized)
    max_val = np.max(image_standardized)
    image_normalized = (image_standardized - min_val) / (max_val - min_val)

    return image_normalized
    
def make_square_with_mirror_padding(image):
    height, width = image.shape[:2]
    if height == width:
        return image
    if height > width:
        difference = height - width
        left_side = image[:, :difference // 2]
        right_side = image[:, -difference // 2:]
        square_image = np.hstack([np.fliplr(left_side), image, np.fliplr(right_side)])
    elif width > height:
        difference = width - height
        top_part = image[:difference // 2, :]
        bottom_part = image[-difference // 2:, :]
        square_image = np.vstack([np.flipud(top_part), image, np.flipud(bottom_part)])
    
    return square_image

def random_rotation(image, max_rotation=45):
    """ Randomly rotates the image within a range of degrees. """
    rotation_degrees = random.uniform(-max_rotation, max_rotation)
    return transform.rotate(image, rotation_degrees)

def random_shift(image, max_shift=0.2):
    """ Applies a random shift to the image. """
    shift_x = np.random.uniform(-max_shift, max_shift) * image.shape[1]
    shift_y = np.random.uniform(-max_shift, max_shift) * image.shape[0]
    translation_matrix = transform.AffineTransform(translation=(shift_x, shift_y))
    return transform.warp(image, translation_matrix)

def random_zoom(image, max_zoom=0.2):
    """ Applies a random zoom to the image. """
    zx, zy = np.random.uniform(1-max_zoom, 1+max_zoom, 2)
    zoom_matrix = transform.AffineTransform(scale=(zx, zy))
    return transform.warp(image, zoom_matrix)

def horizontal_flip(image, flip_prob=0.5):
    """ Flips the image horizontally with a probability. """
    if random.random() < flip_prob:
        return np.fliplr(image)
    return image

def vertical_flip(image, flip_prob=0.5):
    """ Flips the image vertically with a probability. """
    if random.random() < flip_prob:
        return np.flipud(image)
    return image





def rotar_con_espejo(imagen, angulo):
    angulo = random.uniform(-angulo, angulo)

    # Obtener dimensiones de la imagen
    alto, ancho = imagen.shape[:2]

    # Calcular el centro de la imagen
    centro = (ancho // 2, alto // 2)

    # Obtener la matriz de rotación
    matriz_rotacion = cv.getRotationMatrix2D(centro, angulo, 1.0)

    # Rotar la imagen
    imagen_rotada = cv.warpAffine(imagen, matriz_rotacion, (ancho, alto))

    # Crear una máscara para identificar los espacios negros
    mascara = cv.cvtColor(imagen_rotada, cv.COLOR_BGR2GRAY)
    _, mascara = cv.threshold(mascara, 1, 255, cv.THRESH_BINARY)

    # Invertir la máscara para obtener los espacios negros
    mascara_inv = cv.bitwise_not(mascara)

    # Usar la máscara invertida para aplicar el efecto espejo
    imagen_espejo = cv.flip(imagen, 1)  # Efecto espejo horizontal
    imagen_espejo_rotada = cv.warpAffine(imagen_espejo, matriz_rotacion, (ancho, alto))
    imagen_final = cv.bitwise_and(imagen_espejo_rotada, imagen_espejo_rotada, mask=mascara_inv)

    # Combinar la imagen rotada con la imagen de espejo en los espacios negros
    imagen_final = cv.bitwise_or(imagen_rotada, imagen_final)
    return imagen_final









def pad_3D_img(img, padding=[256,256,256]):
    x, y, z = img.shape
    assert x <= padding[0] and y <= padding[1] and z <= padding[2]

    pad_n = max(0, (padding[0] - x) // 2)
    pad_m = max(0, (padding[1] - y) // 2)
    pad_d = max(0, (padding[2] - z) // 2)

    pad_width = (
        (pad_n, pad_n + (padding[0] - x) % 2), 
        (pad_m, pad_m + (padding[1] - y) % 2), 
        (pad_d, pad_d + (padding[2] - z) % 2)
    )

    padded_img = np.pad(img, pad_width, mode='constant', constant_values=0)
    return padded_img

def pad_2D_img(img, padding=[256,256]):
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
        pad_width = pad_width + ((0,0),)

    padded_img = np.pad(img, pad_width, mode='constant', constant_values=0)
    return padded_img

def remove_padding(padded_img, original_shape):
    if padded_img.ndim == 3:
        x, y, _ = original_shape
    else:
        x, y = original_shape

    pad_n = max(0, (padded_img.shape[0] - x) // 2)
    pad_m = max(0, (padded_img.shape[1] - y) // 2)

    if padded_img.ndim == 3:
        return padded_img[pad_n:pad_n+x, pad_m:pad_m+y, :]
    else:
        return padded_img[pad_n:pad_n+x, pad_m:pad_m+y]

def get_2D_slides_from_3D_volume(img_3D, start_slide, end_slide, axis=0):
    img_2D_list = []
    for  slide in range(start_slide, end_slide):
        if axis == 0:
            img_2D = img_3D[slide:slide+1]
        elif axis == 1:
            img_2D = img_3D[:, slide:slide+1]
        elif axis == 2:
            img_2D = img_3D[:, :, slide:slide+1]

        img_2D_list.append(np.squeeze(img_2D))
    return img_2D_list




def reduce_3D_images_batch(img_batch, scaling_factor=2, channels_first=False):
    if not channels_first:
        img_batch = np.transpose(img_batch, (0,4,1,2,3))

    """channels should be axis = 1"""
    new_shape = np.array(img_batch.shape[2:]) // scaling_factor
    img_batch_resized = np.zeros((*img_batch.shape[:2], *new_shape), dtype=img_batch.dtype)
    for i in range(img_batch.shape[0]):
        for c in range(img_batch.shape[1]):
            img_batch_resized[i, c] = zoom(img_batch[i,c], zoom=1/scaling_factor, order=0)

    if not channels_first:
        img_batch_resized = np.transpose(img_batch_resized, (0,2,3,4,1))
    return img_batch_resized



def extract_layer(array, layer, axis):
    if axis < 0 or axis >= array.ndim:
        raise ValueError("La dimensión m es inválida para el arreglo dado.")
    
    if layer < 0 or layer >= array.shape[axis]:
        raise ValueError("El índice n es inválido para la dimensión especificada.")
    
    slicer = [slice(None)] * array.ndim
    slicer[axis] = layer
    
    return array[tuple(slicer)]









def create_index_square_table_size_n(nb_elements):
    numbers = list(range(nb_elements))
    columns = int(np.ceil(np.sqrt(nb_elements)))
    rows = int(np.ceil(nb_elements / columns))

    # Create an array with extra spaces filled with None
    padded_array = np.full(rows * columns, None)
    padded_array[:nb_elements] = numbers

    # Reshape the array into the desired table shape
    return padded_array.reshape(rows, columns)

def create_index_rectangular_table_size_n(nb_elements, rows=None, cols=None):
    if rows is not None and rows < 0:
        rows = None
    if cols is not None and cols < 0:
        cols = None
    if rows is not None and cols is not None:
        print("Only specify either rows or cols, not both.")
        return None
    if rows is None and cols is None:
        print("Specify either rows or cols.")
        return None
    if rows:
        cols = (nb_elements + rows - 1) // rows
    else:
        rows = (nb_elements + cols - 1) // cols 

    # Create an array with extra spaces filled with None
    padded_array = np.full(rows * cols, None)
    padded_array[:nb_elements] = list(range(nb_elements))
    return padded_array.reshape(rows, cols)


def concat_2Dimages_into_rectangular_table(images, custom_disposition=None):
    if custom_disposition is None:
        _table = create_index_square_table_size_n(len(images))
    elif type(custom_disposition) == tuple:
        _table = create_index_rectangular_table_size_n(len(images), rows=custom_disposition[0], cols=custom_disposition[1])
    else:
        _table = custom_disposition
    imgs_dim = images[0].shape
    if len(imgs_dim) == 2:
        full_img = np.zeros((imgs_dim[0] * _table.shape[0], imgs_dim[1] * _table.shape[1]), dtype=np.float32)
    else:
        full_img = np.zeros((imgs_dim[0] * _table.shape[0], imgs_dim[1] * _table.shape[1], imgs_dim[2]), dtype=np.float32)

    for i in range(_table.shape[0]):
        for j in range(_table.shape[1]):
            if _table[i, j] is not None:
                full_img[i * imgs_dim[0]: (i + 1) * imgs_dim[0], j * imgs_dim[1]: (j + 1) * imgs_dim[1]] = images[_table[i, j]]
    return full_img

def concat_3Dimages_into_square_table(images):
    _table = create_index_square_table_size_n(len(images))
    imgs_dim = images[0].shape
    if len(imgs_dim) == 3:
        full_img = np.zeros((imgs_dim[0] * _table.shape[0], imgs_dim[1] * _table.shape[1], imgs_dim[2]), dtype=np.float32)
    else:
        full_img = np.zeros((imgs_dim[0] * _table.shape[0], imgs_dim[1] * _table.shape[1], imgs_dim[2], imgs_dim[3]), dtype=np.float32)

    for i in range(_table.shape[0]):
        for j in range(_table.shape[1]):
            if _table[i, j] is not None:
                full_img[i * imgs_dim[0]: (i + 1) * imgs_dim[0], j * imgs_dim[1]: (j + 1) * imgs_dim[1]] = images[_table[i, j]]
    return full_img


# last added
def get_3Dimg_list_plane(img3D_list, layer, mooving_plane, transpose=False, flip_ax=-1, subtitle_space=20):
    img2D_list = []
    for img3D in img3D_list:
        slice_tuple = [slice(None)] * img3D.ndim
        slice_tuple[mooving_plane] = layer  
        img2D = img3D[tuple(slice_tuple)] 
        if transpose:
            img2D = np.transpose(img2D, (1, 0) + tuple(range(2, img2D.ndim)))
        if flip_ax >=0:
            img2D = np.flip(img2D, axis=flip_ax)
        if subtitle_space is not None:
            if img2D.ndim == 2:
                img2D = np.pad(img2D, ((subtitle_space, 0), (0, 0)), mode='constant', constant_values=0)
            else:
                img2D = np.pad(img2D, ((subtitle_space, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
        img2D_list.append(img2D)
    return img2D_list 


def imgshow_3D_list(name, img3D_list, current_plane=108, cmap='gray', mooving_plane=2, transpose=True, flip_ax=0, show=True, range_values=[None, None], subimg_titles=None, subtitle_space=20, subtitle_help=(20,10), figsize=(10,8), custom_disposition=None, sliders=[], save_dict=None):
    # current_plane = 113
    """Can plot 3D RGB images"""
    # Crear una figura y ejes para la imagen
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.15)

    # Mostrar la imagen inicial
    img2D_list = get_3Dimg_list_plane(img3D_list, current_plane, mooving_plane, transpose=transpose, flip_ax=flip_ax, subtitle_space=subtitle_space)
    _full_img = concat_2Dimages_into_rectangular_table(img2D_list, custom_disposition=custom_disposition).astype(img3D_list[0].dtype)
    img = ax.imshow(_full_img, cmap=cmap, vmin=range_values[0], vmax=range_values[1])

    # Crear un control deslizante para cambiar de plano
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, f'layer {mooving_plane}', 0, img3D_list[0].shape[mooving_plane] - 1, valinit=current_plane, valstep=1)

    sliders.append(slider)
    
    # Función para actualizar la imagen cuando se cambia el valor del slider
    def update(val):
        current_plane = int(slider.val)
        img2D_list = get_3Dimg_list_plane(img3D_list, current_plane, mooving_plane, transpose=transpose, flip_ax=flip_ax, subtitle_space=subtitle_space)
        _full_img = concat_2Dimages_into_rectangular_table(img2D_list, custom_disposition=custom_disposition).astype(img3D_list[0].dtype)
        img.set_data(_full_img)
        fig.canvas.draw_idle()

    ax.set_title(name)
    slider.on_changed(update)
    ax.axis('off')

    plt.rcParams['font.family'] = 'monospace'
    if subimg_titles is not None:
        if custom_disposition is None:
            custom_disposition = create_index_square_table_size_n(len(subimg_titles))
        elif type(custom_disposition) == tuple:
            custom_disposition = create_index_rectangular_table_size_n(len(subimg_titles), rows=custom_disposition[0], cols=custom_disposition[1])
        fontsize= 12
        font_size_inch = fontsize / 72  
        font_size_pixels = int(np.ceil(font_size_inch * fig.dpi ))

        img_size = img2D_list[0].shape
        for row in range(custom_disposition.shape[0]):
            for col in range(custom_disposition.shape[1]):
                if custom_disposition[row, col] is not None:
                    # print(((len(subimg_titles[custom_disposition[row, col]]))*font_size_pixels)//2)
                    init_center_x = img_size[1]//2 - ((len(subimg_titles[custom_disposition[row, col]]))*font_size_pixels)//2
                    pos_x = init_center_x + col*img_size[1] + subtitle_help[0]
                    pos_y = font_size_pixels + row*img_size[0] + subtitle_help[1]
                    ax.text(pos_x, pos_y, subimg_titles[custom_disposition[row, col]], color='white',
                            fontdict={'family': 'monospace', 'weight': 'normal', 'size': fontsize})
    if save_dict is not None:
        plt.savefig(save_dict["path_name"], dpi=save_dict["dpi"], bbox_inches='tight')
        
    if show:
        plt.colorbar(img)
        plt.show()