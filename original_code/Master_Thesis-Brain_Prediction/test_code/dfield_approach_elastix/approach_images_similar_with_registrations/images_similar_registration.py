import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../../'))
dfield_approach_elastix_path = os.path.abspath(os.path.join(current_path, '../'))
sys.path.append(base_path)
sys.path.append(dfield_approach_elastix_path)

import numpy as np
import pandas as pd
from IPython.display import display


import project_images_loader as pil
import utils.util as util
import utils.nifti_functions as nfc
import utils.functions as fc

import constants as cts
import dfield_applier as dfield_applier
from approach_images_similar.similarity_tables_reader import SimilarityTablesReader
from tqdm import tqdm

np.set_printoptions(precision=4)

import itk
    
PARAMETERS_FILES_PATH = f"{current_path}/registrations_parameters_files"
def get_obtained_pfile_path_name_list(id_target, id_data, reg_type):
    obtained_pfile_path_name_list = []
    obtained_pfile_path_name_list.append(f"{PARAMETERS_FILES_PATH}/{id_target}_{id_data}_{cts.PARAMETERS_FILE_RIGID}.txt")
    if reg_type in [cts.PARAMETERS_FILE_AFFINE, cts.PARAMETERS_FILE_BSPLINE]:
        obtained_pfile_path_name_list.append(f"{PARAMETERS_FILES_PATH}/{id_target}_{id_data}_{cts.PARAMETERS_FILE_AFFINE}.txt")
    if reg_type == cts.PARAMETERS_FILE_BSPLINE:
        obtained_pfile_path_name_list.append(f"{PARAMETERS_FILES_PATH}/{id_target}_{id_data}_{cts.PARAMETERS_FILE_BSPLINE}.txt")
    return obtained_pfile_path_name_list


def register_dfield(id_moving, affine, transform_parameters):
    moving_dfield, dfield_aff = nfc.load_nifti(cts.get_image_path_name(id_moving, cts.PFILE_0038, cts.INAME_DFIELD))
    moving_dfield = moving_dfield[:,:,:,0,:]

    dfield_reg_x = itk.transformix_filter(nfc.numpy2itk(moving_dfield[:,:,:,0], affine), transform_parameters)
    dfield_reg_y = itk.transformix_filter(nfc.numpy2itk(moving_dfield[:,:,:,1], affine), transform_parameters)
    dfield_reg_z = itk.transformix_filter(nfc.numpy2itk(moving_dfield[:,:,:,2], affine), transform_parameters)

    dfield_reg_x = itk.array_from_image(dfield_reg_x)
    dfield_reg_y = itk.array_from_image(dfield_reg_y)
    dfield_reg_z = itk.array_from_image(dfield_reg_z)

    dfield_registered = np.stack([dfield_reg_x, dfield_reg_y, dfield_reg_z], axis=3)
    dfield_registered = np.transpose(dfield_registered, (2, 1, 0, 3)) 
    dfield_registered = np.expand_dims(dfield_registered, axis=-2)

    return dfield_registered, dfield_aff



# def register_images(id_target, id_moving, path_img_reg, path_seg_reg, path_dfield_reg):
def save_registration_parameters_file(id_target, id_moving, obtained_pfile_path_name_list, reg_type=False):
    fixed_image = itk.imread(pil.get_registered_to_template_image_path_name(3, id_target, pil.REGISTRATION_TO_MNI, absolute=True), itk.F)
    moving_image = itk.imread(pil.get_registered_to_template_image_path_name(3, id_moving, pil.REGISTRATION_TO_MNI, absolute=True), itk.F)

    fixed_mask = itk.imread(pil.get_brain_mask_path_name(4, id_target, pil.REGISTRATION_TO_MNI, absolute=True), itk.UC)
    moving_mask = itk.imread(pil.get_brain_mask_path_name(3, id_target, pil.REGISTRATION_TO_MNI, absolute=True), itk.UC)

    param_obj_bspline = itk.ParameterObject.New()
    param_obj_bspline.AddParameterFile(cts.get_parameter_file_path_name(cts.PFILE_0038, cts.PARAMETERS_FILE_RIGID))
    if reg_type in [cts.PARAMETERS_FILE_AFFINE, cts.PARAMETERS_FILE_BSPLINE]:
        param_obj_bspline.AddParameterFile(cts.get_parameter_file_path_name(cts.PFILE_0038, cts.PARAMETERS_FILE_AFFINE))
    
    if reg_type == cts.PARAMETERS_FILE_BSPLINE:
        param_obj_bspline.AddParameterFile(cts.get_parameter_file_path_name("0038-FAST", cts.PARAMETERS_FILE_BSPLINE))


    print("Registration started...")
    reg_img, transform_parameters = itk.elastix_registration_method(fixed_image, moving_image,
                                                                            parameter_object=param_obj_bspline,
                                                                            moving_mask=moving_mask,
                                                                            fixed_mask=fixed_mask,
                                                                            log_to_console=False)
    

    transform_parameters.WriteParameterFiles(transform_parameters, obtained_pfile_path_name_list)
    return np.transpose(itk.array_from_image(reg_img), (2, 1, 0)) 


def transform_dfield_from_parameters_file(id_moving, obtained_pfile_path_name_list):
    param_obj = itk.ParameterObject.New()
    for obtained_pfile_path_name in obtained_pfile_path_name_list:
        param_obj.AddParameterFile(obtained_pfile_path_name)

    moving_image_itk  = itk.imread(pil.get_registered_to_template_image_path_name(3, id_moving, pil.REGISTRATION_TO_MNI, absolute=True), itk.F)
    reg_dfield, dfield_aff = register_dfield(id_moving, moving_image_itk, param_obj)
    return reg_dfield, dfield_aff


def transform_segmentation_from_parameters_file(id_moving, obtained_pfile_path_name_list, simple=False):
    param_obj = itk.ParameterObject.New()
    for obtained_pfile_path_name in obtained_pfile_path_name_list:
        param_obj.AddParameterFile(obtained_pfile_path_name)

    param_obj.SetParameter('FinalBSplineInterpolationOrder','0')

    seg_itk = itk.imread(pil.get_segmentation_path_name(3, id_moving, simple=simple, absolute=True), itk.US)
    reg_seg = itk.transformix_filter(seg_itk, param_obj)
    return np.transpose(itk.array_from_image(reg_seg), (2, 1, 0)) 


def create_and_save_best_pfiles(id_target, sim_metric ="ssim", nb_images=10, reg_type=cts.PARAMETERS_FILE_AFFINE, show_metrics=False):
    # load similarity table
    matchin_ids = pil.get_matching_ids()
    sim_table_reder = SimilarityTablesReader(matchin_ids)
    patient_sim_table = sim_table_reder.get_sorted_similarity_table(id_target, sim_metric, nb_images)

    # ---- just to show the metrics
    if show_metrics:
        img_target, _ = nfc.load_nifti(pil.get_registered_to_template_image_path_name(3, id_target, registration_to=pil.REGISTRATION_TO_MNI, absolute=True))
        mask_target, _ = nfc.load_nifti(pil.get_segmentation_path_name(3, id_target, simple=True, absolute=True))
        img_target = util.skull_stripping_and_normalize(img_target, mask_target, remove_negatives=False) 
    # ---- end just to show the metrics
    c = 0
    for index, row in patient_sim_table.iterrows():
        print(c+1)
        obtained_pfile_path_name_list = get_obtained_pfile_path_name_list(id_target, row['id'], reg_type)
        img_data_reg = save_registration_parameters_file(id_target, row['id'], obtained_pfile_path_name_list, reg_type)
        
        # ---- just to show the metrics
        if show_metrics:
            print("Best metrics: ", row["id"], row[sim_metric], row["weight"])
            img_data, _ = nfc.load_nifti(pil.get_registered_to_template_image_path_name(3, row["id"], registration_to=pil.REGISTRATION_TO_MNI, absolute=True))
            mask_data, _ = nfc.load_nifti(pil.get_segmentation_path_name(3, row["id"], simple=True, absolute=True))
            img_data = util.skull_stripping_and_normalize(img_data, mask_data, remove_negatives=False) 

            seg_data_reg = transform_segmentation_from_parameters_file(row['id'], obtained_pfile_path_name_list, simple=True)
            img_data_reg = util.skull_stripping_and_normalize(img_data_reg, seg_data_reg, remove_negatives=False) 

            metrics_reg = util.get_metrics_dict(img_target, img_data_reg)
            print("Registered Metrics: ", metrics_reg)
        c+=1
        # ---- end just to show the metrics


def create_mean_dfield_with_obtained_pfiles(id_target, sim_metric="ssim", nb_images=10, reg_type=cts.PARAMETERS_FILE_AFFINE):
    matchin_ids = pil.get_matching_ids()
    sim_table_reder = SimilarityTablesReader(matchin_ids)
    patient_sim_table = sim_table_reder.get_sorted_similarity_table(id_target, sim_metric, nb_images)

    dfield_sum = None

    progress_bar = tqdm(total=nb_images)
    for index, row in patient_sim_table.iterrows():
        id_value, weight = row['id'], row['weight']

        obtained_pfile_path_name_list = get_obtained_pfile_path_name_list(id_target, row['id'], reg_type)
        _dfield, dfield_aff = transform_dfield_from_parameters_file(id_value, obtained_pfile_path_name_list)

        if dfield_sum is None:
            dfield_sum = _dfield * weight
        else:
            dfield_sum += _dfield * weight
        progress_bar.update(1)
    progress_bar.close()

    return dfield_sum.astype(np.float32), dfield_aff


def save_mean_dfields_with_obtained_pfiles(id_target, sim_metric="ssim", nb_images=10, reg_type=cts.PARAMETERS_FILE_AFFINE):
    dfield_mean, affine = create_mean_dfield_with_obtained_pfiles(id_target, sim_metric, nb_images, reg_type)
    dfield_path_name = cts.get_similar_registered_mean_dfield_path_name(id_target, reg_type, sim_metric.upper(), nb_images)
    nfc.save_nifti(dfield_mean, affine, dfield_path_name)


def do_all_steps(id_indexs, sim_metric="ssim", nb_images_list=[1,10], reg_type_list=[cts.PARAMETERS_FILE_AFFINE]):
    matchin_ids = pil.get_matching_ids()
    # # compute mean dfield
    for id_index in id_indexs:
        id = matchin_ids[id_index] 
        print(f"-------- ID: {id}")
        for nb_images in nb_images_list:
            print(f"---- Best {nb_images}: ")
            for reg_type in reg_type_list:
                create_and_save_best_pfiles(id, sim_metric, nb_images, reg_type, show_metrics=True)
                save_mean_dfields_with_obtained_pfiles(id, sim_metric, nb_images, reg_type)
                save_mean_dfields_with_obtained_pfiles(id, sim_metric, nb_images, cts.PARAMETERS_FILE_AFFINE)

    
# do_all_steps([301, 303, 304, 305], sim_metric="ssim", nb_images_list=[1, 10], reg_type_list=[cts.PARAMETERS_FILE_BSPLINE])