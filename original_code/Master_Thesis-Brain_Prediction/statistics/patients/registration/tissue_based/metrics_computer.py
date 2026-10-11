

import numpy as np
import SimpleITK as sitk


def compute_dfield_jacobian(sitk_displacement_field):
    jacobian_filter = sitk.DisplacementFieldJacobianDeterminantFilter()
    jacobian_filter.SetUseImageSpacing(True)
    jacobian_image = jacobian_filter.Execute(sitk_displacement_field)

    # or simply
    # jacobian_image = sitk.DisplacementFieldJacobianDeterminant(sitk_displacement_field)
    return sitk.GetArrayFromImage(jacobian_image)

def compute_dfield_divergence(field):   
    div_x = np.gradient(field[:, :, :, 0], axis=0)
    div_y = np.gradient(field[:, :, :, 1], axis=1)
    div_z = np.gradient(field[:, :, :, 2], axis=2)
    
    div = div_x + div_y + div_z
    
    return div

def compute_dfield_magnitude(field):   
    mag = np.sqrt(field[:, :, :, 0]**2 + field[:, :, :, 1]**2 + field[:, :, :, 2]**2)
    return mag
