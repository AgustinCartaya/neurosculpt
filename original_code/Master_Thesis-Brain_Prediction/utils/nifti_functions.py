import nibabel as nib
import numpy as np
from . import functions as fc
# import itk
import SimpleITK as sitk


# ------------ LOAD IMAGES
def load_nifti(path_name, transpose=False):
    imag_nifti = nib.load(path_name)
    img_data = imag_nifti.get_fdata()
    if transpose:
        img_data = np.transpose(img_data, (1, 0, 2))
    return img_data, (imag_nifti.affine, imag_nifti.header)

# def load_nifti_itk(path_name, itk_format=itk.F):
#     itk_object = itk.imread(path_name, itk_format)
#     return itk.array_from_image(itk_object), itk_object

def load_image_sitk(path_name):
    img_sitk = sitk.ReadImage(path_name)
    return sitk.GetArrayFromImage(img_sitk), img_sitk


# ------------ DATA TYPE CONVERSIONS 
def nifti2cv(nifti, layer, color_image=False):  
    return fc.img_float2unit8(nifti[:,:, layer].T, color_image=color_image)

# def numpy2itk(np_array, itk_object):
#     itk_image = itk.image_view_from_array(np_array)
#     itk_image.CopyInformation(itk_object)
#     return itk_image

# ------------ SAVE IMAGES
def save_nifti(image_np, affine, img_path_name, transpose=False):
    if transpose:
        image_np = np.transpose(image_np, (1, 0, 2)) 

    img_nifti = nib.Nifti1Image(image_np, affine=affine[0], header=affine[1])
    nib.save(img_nifti, img_path_name)


# def save_nifti_itk(image_np, itk_object, img_path_name, compression=True):
#     itk_image = numpy2itk(image_np, itk_object)
#     itk.imwrite(itk_image, img_path_name, compression=compression)

    


