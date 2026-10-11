import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../../'))
sys.path.append(base_path)

import numpy as np
import pandas as pd

import project_images_loader as pil
import utils.util as util
import utils.nifti_functions as nfc
import utils.functions as fc

import constants as cts
import utils.dfield_applier as dfield_applier
# from approach_images_similar.similarity_tables_reader import SimilarityTablesReader
features_loader_tissue_based = fc.load_module_from_path('features_loader_tissue_based', os.path.join(base_path, 'statistics/patients/inter_patient/tissue_based/features_loader.py'))

from tqdm import tqdm

np.set_printoptions(precision=4)

import itk
    




def save_registration_parameters_file(id_target, id_moving, obtained_pfile_path_name_list, reg_type=False):
    fixed_image = itk.imread(pil.get_registered_to_template_image_path_name(3, id_target, pil.REGISTRATION_TO_MNI, absolute=True), itk.F)
    moving_image = itk.imread(pil.get_registered_to_template_image_path_name(3, id_moving, pil.REGISTRATION_TO_MNI, absolute=True), itk.F)

    fixed_mask = itk.imread(pil.get_brain_mask_path_name(4, id_target, pil.REGISTRATION_TO_MNI, absolute=True), itk.UC)
    moving_mask = itk.imread(pil.get_brain_mask_path_name(3, id_target, pil.REGISTRATION_TO_MNI, absolute=True), itk.UC)

    param_obj_bspline = itk.ParameterObject.New()
    param_obj_bspline.AddParameterFile(cts.get_parameter_file_path_name(cts.PARAMETERS_FILE_RIGID))
    if reg_type in [cts.PARAMETERS_FILE_AFFINE, cts.PARAMETERS_FILE_BSPLINE, cts.PARAMETERS_FILE_BSPLINE_FAST]:
        param_obj_bspline.AddParameterFile(cts.get_parameter_file_path_name(cts.PARAMETERS_FILE_AFFINE))
    if reg_type == cts.PARAMETERS_FILE_BSPLINE:
        param_obj_bspline.AddParameterFile(cts.get_parameter_file_path_name(cts.PARAMETERS_FILE_BSPLINE))
    if reg_type == cts.PARAMETERS_FILE_BSPLINE_FAST:
        param_obj_bspline.AddParameterFile(cts.get_parameter_file_path_name(cts.PARAMETERS_FILE_BSPLINE_FAST))


    print("Registration started...")
    reg_img, transform_parameters = itk.elastix_registration_method(fixed_image, moving_image,
                                                                            parameter_object=param_obj_bspline,
                                                                            moving_mask=moving_mask,
                                                                            fixed_mask=fixed_mask,
                                                                            log_to_console=False)
    

    transform_parameters.WriteParameterFiles(transform_parameters, obtained_pfile_path_name_list)
    # return np.transpose(itk.array_from_image(reg_img), (2, 1, 0)) 





def numpy2itk(np_array, itk_object):
    itk_image = itk.image_view_from_array(np_array)
    itk_image.CopyInformation(itk_object)
    return itk_image

def register_dfield(id_moving, affine, transform_parameters):
    moving_dfield, dfield_aff = nfc.load_nifti(pil.get_elastix_registered_image_path_name(id_moving, type_img="DFIELD", absolute=True))
    moving_dfield = moving_dfield[:,:,:,0,:]

    dfield_reg_x = itk.transformix_filter(numpy2itk(moving_dfield[:,:,:,0], affine), transform_parameters)
    dfield_reg_y = itk.transformix_filter(numpy2itk(moving_dfield[:,:,:,1], affine), transform_parameters)
    dfield_reg_z = itk.transformix_filter(numpy2itk(moving_dfield[:,:,:,2], affine), transform_parameters)

    dfield_reg_x = itk.array_from_image(dfield_reg_x)
    dfield_reg_y = itk.array_from_image(dfield_reg_y)
    dfield_reg_z = itk.array_from_image(dfield_reg_z)

    dfield_registered = np.stack([dfield_reg_x, dfield_reg_y, dfield_reg_z], axis=3)
    dfield_registered = np.transpose(dfield_registered, (2, 1, 0, 3)) 
    dfield_registered = np.expand_dims(dfield_registered, axis=-2)

    return dfield_registered, dfield_aff


def register_image_and_mask(id_moving, transform_parameters):
    _, img_aff = nfc.load_nifti(pil.get_registered_to_template_image_path_name(3, id_moving, registration_to=pil.REGISTRATION_TO_MNI, absolute=True))
    _, seg_aff = nfc.load_nifti(pil.get_segmentation_path_name(3, id_moving, nb_tissues=96, absolute=True))
    moving_img  = itk.imread(pil.get_registered_to_template_image_path_name(3, id_moving, registration_to=pil.REGISTRATION_TO_MNI, absolute=True), itk.F)
    moving_seg  = itk.imread(pil.get_segmentation_path_name(3, id_moving, nb_tissues=96, absolute=True), itk.UC)

    img_reg = itk.transformix_filter(moving_img, transform_parameters)

    transform_parameters.SetParameter('FinalBSplineInterpolationOrder','0')
    seg_reg = itk.transformix_filter(moving_seg, transform_parameters)

    img_reg = itk.array_from_image(img_reg)
    seg_reg = itk.array_from_image(seg_reg)

    img_reg = np.transpose(img_reg, (2, 1, 0)) 
    seg_reg = np.transpose(seg_reg, (2, 1, 0)) 

    return (img_reg, img_aff), (seg_reg, seg_aff)


def load_reg_param_files(obtained_pfile_path_name_list):
    param_obj = itk.ParameterObject.New()
    for obtained_pfile_path_name in obtained_pfile_path_name_list:
        param_obj.AddParameterFile(obtained_pfile_path_name)
    return param_obj


def transform_dfield_from_parameters_file(id_moving, obtained_pfile_path_name_list):
    param_obj = itk.ParameterObject.New()
    for obtained_pfile_path_name in obtained_pfile_path_name_list:
        param_obj.AddParameterFile(obtained_pfile_path_name)

    moving_image_itk  = itk.imread(pil.get_registered_to_template_image_path_name(3, id_moving, pil.REGISTRATION_TO_MNI, absolute=True), itk.F)
    reg_dfield, dfield_aff = register_dfield(id_moving, moving_image_itk, param_obj)
    return reg_dfield, dfield_aff



# def create_and_save_best_pfiles(id_target, patient_sim_table, reg_type=cts.PARAMETERS_FILE_AFFINE, show_metrics=False):
#     # ---- just to show the metrics
#     if show_metrics:
#         img_target, _ = nfc.load_nifti(pil.get_registered_to_template_image_path_name(3, id_target, registration_to=pil.REGISTRATION_TO_MNI, absolute=True))
#         mask_target, _ = nfc.load_nifti(pil.get_segmentation_path_name(3, id_target, nb_tissues=3, absolute=True))
#         img_target = util.skull_stripping_and_normalize(img_target, mask_target, remove_negatives=False) 
#     # ---- end just to show the metrics
#     c = 0
#     for index, row in patient_sim_table.iterrows():
#         print(c+1)
#         obtained_pfile_path_name_list = get_obtained_pfile_path_name_list(id_target, row['id'], reg_type)
#         img_data_reg = save_registration_parameters_file(id_target, row['id'], obtained_pfile_path_name_list, reg_type)
        
#         # ---- just to show the metrics
#         if show_metrics:
#             print("Best metrics: ", row["id"], row["dice_mean"], row["weight"])
#             img_data, _ = nfc.load_nifti(pil.get_registered_to_template_image_path_name(3, row["id"], registration_to=pil.REGISTRATION_TO_MNI, absolute=True))
#             mask_data, _ = nfc.load_nifti(pil.get_segmentation_path_name(3, row["id"], nb_tissues=3, absolute=True))
#             img_data = util.skull_stripping_and_normalize(img_data, mask_data, remove_negatives=False) 

#             seg_data_reg = transform_segmentation_from_parameters_file(row['id'], obtained_pfile_path_name_list, simple=True)
#             img_data_reg = util.skull_stripping_and_normalize(img_data_reg, seg_data_reg, remove_negatives=False) 

#             metrics_reg = util.get_metrics_dict(img_target, img_data_reg)
#             print("Registered Metrics: ", metrics_reg)
#         c+=1
#         # ---- end just to show the metrics


# def create_mean_dfield_with_obtained_pfiles(id_target, sim_metric="dice_mean", nb_images=5, reg_type=cts.PARAMETERS_FILE_AFFINE):
#     matchin_ids = pil.get_matching_ids()
#     sim_table_reder = SimilarityTablesReader(matchin_ids)
#     patient_sim_table = sim_table_reder.get_sorted_similarity_table(id_target, sim_metric, nb_images)

#     dfield_sum = None

#     progress_bar = tqdm(total=nb_images)
#     for index, row in patient_sim_table.iterrows():
#         id_value, weight = row['id'], row['weight']

#         obtained_pfile_path_name_list = get_obtained_pfile_path_name_list(id_target, row['id'], reg_type)
#         _dfield, dfield_aff = transform_dfield_from_parameters_file(id_value, obtained_pfile_path_name_list)

#         if dfield_sum is None:
#             dfield_sum = _dfield * weight
#         else:
#             dfield_sum += _dfield * weight
#         progress_bar.update(1)
#     progress_bar.close()

#     return dfield_sum.astype(np.float32), dfield_aff


# def save_mean_dfields_with_obtained_pfiles(id_target, sim_metric="dice_mean", nb_images=5, reg_type=cts.PARAMETERS_FILE_AFFINE):
#     dfield_mean, affine = create_mean_dfield_with_obtained_pfiles(id_target, sim_metric, nb_images, reg_type)
#     dfield_path_name = cts.get_similar_registered_mean_dfield_path_name(id_target, reg_type, sim_metric.upper(), nb_images)
#     nfc.save_nifti(dfield_mean, affine, dfield_path_name)



def verify_parameters_files_path_name(obtained_pfile_path_name_list):
    all_created = False
    for obtained_pfile_path_name in obtained_pfile_path_name_list:
        all_created = fc.verify_file(obtained_pfile_path_name)
        if not all_created:
            break
    return all_created        

def do_all_steps(ids, sim_metric="dice_mean", nb_images=1, reg_type=cts.PARAMETERS_FILE_BSPLINE):
    matching_ids = pil.get_matching_ids()
    # # compute mean dfield
    for id in ids:
        if fc.verify_folder(cts.get_pateint_similar_images_registration_folder_path(id), create=True):
       
            print(f"-------- ID: {id}")
            print(f"---- Best {nb_images}: ")
            # load the n best images for target image
            dice_mean_table_reder = features_loader_tissue_based.FeaturesLoader(3, matching_ids)
            patient_sim_table = dice_mean_table_reder.get_sorted_similarity_table(id, sim_metric, nb_images, exclude_ids= ids + pil.EXCLUDED_PATIENTS)
            
            for index, row in patient_sim_table.iterrows():
                obtained_pfile_path_name_list = cts.get_obtained_pfile_path_name_list(id, row['id'], reg_type)
                print(row['id'])
                if verify_parameters_files_path_name(obtained_pfile_path_name_list):
                    # step1: save registration parameters file
                    # save_registration_parameters_file(id, row['id'], obtained_pfile_path_name_list, reg_type)
                    # step2: transform dfield from parameters file
                    # reg_dfield, dfield_aff = transform_dfield_from_parameters_file(row['id'], obtained_pfile_path_name_list)
                    # reg_dfield_path_name = cts.get_patient_registered_dfield_path_name(id, row['id'], reg_type)
                    # nfc.save_nifti(reg_dfield, dfield_aff, reg_dfield_path_name)

                    # step2: transform img and seg #porter images
                    params = load_reg_param_files(obtained_pfile_path_name_list)
                    (img_reg, img_aff), (seg_reg, seg_aff) = register_image_and_mask(row['id'], params)
                    reg_img_path_name = cts.get_patient_registered_custom_path_name(id, row['id'], "T1", reg_type)
                    reg_seg_path_name = cts.get_patient_registered_custom_path_name(id, row['id'], "SEG_96", reg_type)
                    nfc.save_nifti(img_reg, img_aff, reg_img_path_name)
                    nfc.save_nifti(seg_reg, seg_aff, reg_seg_path_name)




# val_images = ["08202"] + pil.get_matching_ids()[621:]
val_images = ["17112"]
# do_all_steps(val_images)

do_all_steps(val_images, nb_images=5, reg_type=cts.PARAMETERS_FILE_BSPLINE_FAST)
# do_all_steps(val_images, nb_images=5, reg_type=cts.PARAMETERS_FILE_BSPLINE_FAST)
