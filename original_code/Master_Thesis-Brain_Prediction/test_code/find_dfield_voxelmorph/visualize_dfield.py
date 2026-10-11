import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(base_path)

import subprocess
import numpy as np
import project_images_loader as pil
import utils.nifti_functions as nfc
import nibabel as nib

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importa el módulo necesario para gráficas 3D
from matplotlib.colors import Normalize
# Load the deformation field from the NIfTI file
step = 10
arrow_size = 0.5
id = "00039"
hunt= 3
DFIELD_NAME = "{0}_{1}_DFIELD_VOXELMORPH2.nii.gz"
INAME_REGISTERED = "{0}_{1}_T1_VOXELMORPH2.nii.gz"

deformation_field, _ = nfc.load_nifti(f"{current_path}/images/{DFIELD_NAME.format(id, hunt-3)}")
img_registeres, _ = nfc.load_nifti(f"{current_path}/images/{INAME_REGISTERED.format(id, hunt-3)}")

# Aplica la máscara al campo de deformación
deformation_field[img_registeres == 0] = 0



deformation_field = deformation_field[::step, ::step, ::step,:]
# print(deformation_field.shape)

# Obtiene las dimensiones del campo de deformación
x_dim, y_dim, z_dim, _ = deformation_field.shape

# Crea una cuadrícula de coordenadas para la trama
x = np.arange(0, x_dim)  # Añade un paso de 2
y = np.arange(0, y_dim)  # Añade un paso de 2
z = np.arange(0, z_dim)  # Añade un paso de 2
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Calcula la magnitud del campo de deformación
magnitude = np.linalg.norm(deformation_field[::step, ::step, ::step, :], axis=-1)

# Normaliza la magnitud para que pueda ser utilizada para colorear las flechas
norm = Normalize()
colors = plt.cm.viridis(norm(magnitude.flatten()))


# Traza el campo de deformación
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # Crea un gráfico 3D
ax.quiver(X, Y, Z, deformation_field[..., 0], deformation_field[..., 1], deformation_field[..., 2], length=arrow_size, colors=colors, normalize=True)
plt.show()

















# import sys
# import os

# current_path = os.path.dirname(__file__)
# base_path = os.path.abspath(os.path.join(current_path, '../../'))
# sys.path.append(base_path)

# import subprocess
# import numpy as np
# import project_images_loader as pil
# import utils.nifti_functions as nfc
# import nibabel as nib

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # Importa el módulo necesario para gráficas 3D
# from matplotlib.colors import Normalize

# # Load the deformation field from the NIfTI file
# step = 15
# id = 12082
# deformation_field, _ = nfc.load_nifti(pil.get_dfield_path_name(id, pil.REGISTRATION_TO_HUNT3, pil.DFIELD_METHOD_ELASTIX, absolute=True))
# deformation_field = deformation_field[::step,::step,::step,0,:]
# # print(deformation_field.shape)

# # Obtiene las dimensiones del campo de deformación
# x_dim, y_dim, z_dim, _ = deformation_field.shape

# # Crea una cuadrícula de coordenadas para la trama
# x = np.arange(0, x_dim, step)  # Añade un paso de 2
# y = np.arange(0, y_dim, step)  # Añade un paso de 2
# z = np.arange(0, z_dim, step)  # Añade un paso de 2
# X, Y, Z = np.meshgrid(x, y, z, indexing='ij')


# # Calcula la magnitud del campo de deformación
# magnitude = np.linalg.norm(deformation_field, axis=-1)

# # Normaliza la magnitud para que pueda ser utilizada para colorear las flechas
# # norm = Normalize()
# # colors = plt.cm.viridis(norm(magnitude.flatten()))


# # Traza el campo de deformación
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')  # Crea un gráfico 3D
# # ax.quiver(X, Y, Z, deformation_field[..., 0], deformation_field[..., 1], deformation_field[..., 2], length=5.0, colors=colors, linewidths=0.5)
# ax.quiver(X, Y, Z, deformation_field[..., 0], deformation_field[..., 1], deformation_field[..., 2], length=5.0, linewidths=0.5)
# plt.show()