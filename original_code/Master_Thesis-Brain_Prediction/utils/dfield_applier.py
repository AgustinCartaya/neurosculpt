
import numpy as np

import itk

import tensorflow as tf
# import voxelmorph as vxm

import scipy.ndimage


def apply_deformation_field_elastix(img_path_name, dfield_path_name, binary=False):
    Dimension = 3
    DisplacementFieldType = itk.Image[itk.Vector[itk.F, Dimension], Dimension]

    ImageType = itk.Image[itk.D, Dimension]
    img_reader = itk.ImageFileReader[ImageType].New()
    img_reader.SetFileName(img_path_name)
    img = img_reader.GetOutput()

    dfield_reader = itk.ImageFileReader[DisplacementFieldType].New()
    dfield_reader.SetFileName(dfield_path_name)
    dfield_reader.Update()
    dfield = dfield_reader.GetOutput()

    interpolator = itk.BSplineInterpolateImageFunction[ImageType, itk.D, itk.D].New()
    if binary:
        interpolator.SetSplineOrder(0)
    else:
        interpolator.SetSplineOrder(3)
            
    warpFilter = itk.WarpImageFilter[ImageType, ImageType, DisplacementFieldType].New()
    warpFilter.SetInterpolator(interpolator)
    warpFilter.SetOutputSpacing(dfield.GetSpacing())
    warpFilter.SetOutputOrigin(dfield.GetOrigin())
    warpFilter.SetOutputDirection(dfield.GetDirection())
    warpFilter.SetDisplacementField(dfield)
    warpFilter.SetInput(img)
    warpFilter.Update()

    img_output = warpFilter.GetOutput()
    img_output_np = itk.GetArrayFromImage(img_output)
    return img_output_np.transpose(2, 1, 0)


# def apply_deformation_field_voxelmorph(img, dfield, binary=False):
#     img_tensor = tf.expand_dims(tf.expand_dims(img, axis=-1), axis=0)
#     dfield_tensor = tf.expand_dims(dfield, axis=0)

#     if binary:
#         spatial_transformer = vxm.layers.SpatialTransformer(name='transformer', indexing='ij', interp_method='nearest')
#     else:
#         spatial_transformer = vxm.layers.SpatialTransformer(name='transformer', indexing='ij', interp_method='linear')
        
#     moved_image_tensor = spatial_transformer([img_tensor, dfield_tensor])
#     moved_image = moved_image_tensor[0,:,:,:,0].numpy()
#     return moved_image


def apply_deformation_field_mapping_coords(image, deformation_field, binary=False):
    # Obtener la forma de la imagen y el campo de deformación
    image_shape = image.shape
    deformation_shape = deformation_field.shape[:-1]  # Eliminar la dimensión de los componentes del desplazamiento

    # Asegurarse de que la forma de la imagen y el campo de deformación sean compatibles
    assert image_shape == deformation_shape, "La forma de la imagen y el campo de deformación no coinciden"

    # Transponer el primer eje del campo de deformación
    deformation_field = np.transpose(deformation_field, (3, 0, 1, 2))

    # Generar las coordenadas de malla para la imagen deformada
    coordinates = np.indices(image_shape)
    
    # Calcular las nuevas coordenadas basadas en el campo de deformación
    new_coordinates = coordinates + deformation_field

    # Interpolar la imagen deformada utilizando las nuevas coordenadas
    if binary:
        deformed_image = scipy.ndimage.map_coordinates(image, new_coordinates, order=0, mode='constant')
    else:
        deformed_image = scipy.ndimage.map_coordinates(image, new_coordinates, order=1, mode='constant')

    return deformed_image






