import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(base_path)

import subprocess
import numpy as np
import project_images_loader as pil
import utils.nifti_functions as nfc
import utils.functions as fc
import utils.util_freesurfer_segmentation as util_freesurfer_segmentation
import nibabel as nib
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importa el módulo necesario para gráficas 3D
from matplotlib.colors import Normalize
# Load the deformation field from the NIfTI file

import constants as cts
import neurite as ne


def apply_segmentation(dfield, seg, labels):
    if seg is not None and labels is not None:
        if isinstance(labels, list):
            dfield = util_freesurfer_segmentation.segment_freesurfer_labels(dfield, seg, labels)
        else:
            dfield[seg == 0] = 0
    return dfield

def visualize_3D_dfield(dfield, seg=None, labels=None, step=10, arrow_size=1.0):
    # Aplica la máscara al campo de deformación
    dfield = apply_segmentation(dfield, seg, labels)

    # Reduce el tamaño del campo de deformación
    dfield = dfield[::step, ::step, ::step, :]

    # Obtiene las dimensiones del campo de deformación
    x_dim, y_dim, z_dim, _ = dfield.shape

    # Crea una cuadrícula de coordenadas para la trama
    x = np.arange(0, x_dim)  # Añade un paso de 2
    y = np.arange(0, y_dim)  # Añade un paso de 2
    z = np.arange(0, z_dim)  # Añade un paso de 2
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Calcula la magnitud del campo de deformación
    magnitude = np.linalg.norm(dfield[::step, ::step, ::step, :], axis=-1)

    # Normaliza la magnitud para que pueda ser utilizada para colorear las flechas
    norm = Normalize()
    colors = plt.cm.viridis(norm(magnitude.flatten()))

    # Traza el campo de deformación
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Crea un gráfico 3D
    ax.quiver(X, Y, Z, dfield[..., 0], dfield[..., 1], dfield[..., 2], length=arrow_size, colors=colors, normalize=True)
    plt.show()


def get_dfield_2D_image(dfield2D, width=30):
    fig, axs = ne.plot.flow([dfield2D.squeeze()], width=width, show=False)
    fig.canvas.draw()
    img = np.array(fig.canvas.buffer_rgba())
    plt.close('all')
    return img

def get_usefull_inside_layers_index(img, axis=2):
    init_layer = 0
    end_layer = img.shape[axis]
    for i in range(img.shape[axis]):
        img_layer = fc.get_3Dimg_plane(img, i, axis)
        if np.sum(img_layer) > 0:
            init_layer = i
            break

    for i in range(img.shape[axis]-1, -1, -1):
        img_layer = fc.get_3Dimg_plane(img, i, axis)
        if np.sum(img_layer) > 0:
            end_layer = i
            break
    return init_layer, end_layer
     
def visualize_2D_dfield(dfield, seg=None, labels=None, axis=2, step=10, width=10):
    dfield = apply_segmentation(dfield, seg, labels)
    init_layer, end_layer = get_usefull_inside_layers_index(dfield, axis=axis)
    print(init_layer, end_layer)
    dfield_2DRGB = None
    dfield_components_to_show = [0,1,2]
    dfield_components_to_show.remove(axis)
    for layer in tqdm(range(init_layer, end_layer)):
        dfield_layer = fc.get_3Dimg_plane(dfield, layer, axis)
        dfield_layer = dfield_layer[::step, ::step, dfield_components_to_show]
        dfield_img_flow = get_dfield_2D_image(dfield_layer, width=width)

        if dfield_2DRGB is None:
            img_shape = dfield_img_flow.shape
            dfield_2DRGB = np.zeros((img_shape[0], img_shape[1], dfield.shape[axis], 3), dtype=np.uint8)
        
        dfield_2DRGB[:, :, layer, :] = dfield_img_flow[:, :, :3]

    mean_layer = (init_layer + end_layer)//2
    fc.imgshow_3D("", dfield_2DRGB, current_plane=mean_layer, mooving_plane=2, transpose=True, flip_ax=0)




id = "02354"
pfile = cts.PFILE_0038

dfield, _ = nfc.load_nifti(cts.get_image_path_name(id, pfile, cts.INAME_DFIELD))
dfield = dfield[:,:,:,0,:]
# dfield = np.transpose(dfield, (1, 0, 2, 3))

seg, _ = nfc.load_nifti(cts.get_image_path_name(id, pfile, cts.INAME_SEG_REGISTERED), transpose=False)



# visualize_2D_dfield(dfield, seg, labels=[43, 4], axis=2, step=2, width=30)
# visualize_3D_dfield(dfield, seg, labels=[43, 4], step=2, arrow_size=1.0)

seg_RGB = util_freesurfer_segmentation.seg_to_rgbseg(seg)
seg_RGB = apply_segmentation(seg_RGB, seg, 0)
init_layer, end_layer = get_usefull_inside_layers_index(seg_RGB, axis=2)
seg_RGB = seg_RGB[:,:,init_layer:end_layer,:]
fc.imgshow_3D("", seg_RGB, current_plane=10, mooving_plane=2, transpose=True, flip_ax=0)


# https://elastix.lumc.nl/doxygen/classelastix_1_1DeformationFieldTransform.html