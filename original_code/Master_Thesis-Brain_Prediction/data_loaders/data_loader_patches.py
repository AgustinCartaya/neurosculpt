import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../'))
sys.path.append(base_path)

import numpy as np
import utils.nifti_functions as nfc
import random
import utils.functions as fc
import project_images_loader as pil
import utils.patches_controller as patches_controller
import utils.util_freesurfer_segmentation as ufs
from data_loaders.data_loader_ids import DataLoaderIds

class DataLoaderPatches:

    HUNT3_FOLDER = "HUNT3"
    HUNT4_FOLDER = "HUNT4"
    def __init__(self, image_sufix, batch_size, image_size, patch_size, patch_step, images_onmemory=20, shuffle_patches=False, channels_first=False, background_th = 0.3,
                 shuffle_images=False, nb_patients=-1, shuffle_before_selecting=False, server=True, data_type=np.float32, exclude_ids=None, start_from=0, image_sufix_seg_gmm=None):
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_step = patch_step
        self.images_onmemory = images_onmemory
        self.shuffle_patches = shuffle_patches
        self.channels_first = channels_first
        self.background_th = background_th
        self.data_type = data_type
        self.shuffle_images = shuffle_images
        self.image_sufix_seg_gmm = image_sufix_seg_gmm

        self.ids_loader = DataLoaderIds(image_sufix=image_sufix, 
                                        batch_size=images_onmemory, 
                                        shuffle=shuffle_images, # only to shuffle at the begining 
                                        nb_patients=nb_patients, 
                                        shuffle_before_selecting=shuffle_before_selecting, 
                                        server=server,
                                        exclude_ids=exclude_ids,
                                        start_from=start_from)
        
        self.util_patch_th = np.int32(background_th * np.prod(patch_size[:]))
        self.current_index_batch_images = 0

        # custom patches
        self.patches_indexs = None
        self.usefull_patches = {}
        self.custom_patches = {}
        self.custom_patches_metadata = {}
        self.iter_patches_index = None
        self.iter_patches_index_ids = None

        self.iter_patient_gmm = []
        self.__load_patches()

    def __load_patches(self):
        self.patches_hunt3 = np.empty((0,) + self.patch_size, dtype=self.data_type)
        self.patches_hunt4 = np.empty((0,) + self.patch_size, dtype=self.data_type)
        self.index_ids_patches = np.empty((0,), dtype=np.uint8)
        self.iter_patient_gmm = []

        images_ids = self.ids_loader[self.current_index_batch_images]
        print(f"Loading patches from images: {images_ids}")

        for i, id in enumerate(images_ids):
            # load images
            img_hunt3, _ = nfc.load_nifti(self.ids_loader.get_image_path_name(3, id))
            img_hunt4, _ = nfc.load_nifti(self.ids_loader.get_image_path_name(4, id))

            # extract patches
            _patches_hunt3 = patches_controller.extract_patches(img_hunt3, self.patch_size, self.patch_step)
            _patches_hunt4 = patches_controller.extract_patches(img_hunt4, self.patch_size, self.patch_step)
            
            # extract usefull patches
            _usefull_patches = self.get_usefull_patches(3, id)    
            _patches_hunt3 = _patches_hunt3[_usefull_patches]
            _patches_hunt4 = _patches_hunt4[_usefull_patches]
            self.usefull_patches[id] = _usefull_patches

            # store usefull patches
            self.patches_hunt3 = np.concatenate((self.patches_hunt3, _patches_hunt3), axis=0)
            self.patches_hunt4 = np.concatenate((self.patches_hunt4, _patches_hunt4), axis=0)

            # store index ids
            self.index_ids_patches = np.concatenate((self.index_ids_patches, [i] * len(_patches_hunt3)), axis=0)

            # store gmm
            if self.image_sufix_seg_gmm is not None:
                self.__apped_gmm(img_hunt3, img_hunt4, id)

        self.nb_patches = self.patches_hunt3.shape[0]
        
        self.patches_hunt3 = self.patches_hunt3.astype(self.data_type)
        self.patches_hunt4 = self.patches_hunt4.astype(self.data_type)
        
        self.patches_indexs = np.arange(len(self.patches_hunt3))
        self.__shuffle_patches()

        # custom patches
        for _name, _meta in self.custom_patches_metadata.items():
            self.prepare_custom_patches(_name, _meta["custom_img_sufix"], _meta["data_type"], _meta["img_transform"])

    def __shuffle_patches(self):
        if self.shuffle_patches:
            np.random.shuffle(self.patches_indexs)
            self.patches_hunt3 = self.patches_hunt3[self.patches_indexs]
            self.patches_hunt4 = self.patches_hunt4[self.patches_indexs]
            self.index_ids_patches = self.index_ids_patches[self.patches_indexs]

    def __len__(self):
        return int(np.ceil(self.nb_patches/self.batch_size))

    def __getitem__(self, index):
        if index < 0:
            index = int(np.ceil(self.nb_patches / self.batch_size)) + index

        start = index * self.batch_size
        end = min(start + self.batch_size, self.nb_patches)
        return self.__get_batch(start,end)           

    def __iter__(self):
        for i in range(0, self.nb_patches, self.batch_size):
            yield self.__get_batch(i, min(i + self.batch_size, self.nb_patches))

        self.current_index_batch_images += 1
        if self.current_index_batch_images >= len(self.ids_loader):
            print("Dataset ended, starting from the beginning...")
            self.current_index_batch_images = 0
            if self.shuffle_images:
                self.ids_loader.shuffle_ids()
        self.__load_patches()


    def __get_batch(self, start, end):
        self.iter_patches_index = (start, end)
        self.iter_patches_index_ids = self.index_ids_patches[start:end]

        batch_hunt3 = self.patches_hunt3[start:end]
        batch_hunt4 = self.patches_hunt4[start:end]

        batch_hunt3 = np.expand_dims(batch_hunt3, axis=1)
        batch_hunt4 = np.expand_dims(batch_hunt4, axis=1)
    
        if not self.channels_first:
            batch_hunt3 = np.moveaxis(batch_hunt3, 1, -1)
            batch_hunt4 = np.moveaxis(batch_hunt4, 1, -1)

        return batch_hunt3, batch_hunt4

    def __apped_gmm(self, img_hunt3, img_hunt4, id):
        seg_hunt3, _ = nfc.load_nifti(self.ids_loader.get_custom_image_path_name(3, id, self.image_sufix_seg_gmm))
        seg_hunt4, _ = nfc.load_nifti(self.ids_loader.get_custom_image_path_name(4, id, self.image_sufix_seg_gmm))
        self.iter_patient_gmm.append((ufs.obtain_gmm_elements_using_seg(img_hunt3, seg_hunt3),
                                      ufs.obtain_gmm_elements_using_seg(img_hunt4, seg_hunt4)))
        
    def get_iter_patient_gmm(self):
        if self.image_sufix_seg_gmm is None:
            return None
        return np.array([self.iter_patient_gmm[i] for i in self.iter_patches_index_ids], dtype=np.float32).transpose(1, 2, 0, 3)
         

    def get_usefull_patches(self, hunt, id):  
        mask, _ = nfc.load_nifti(self.ids_loader.get_image_bmask_path_name(hunt, id))  
        mask_patches = patches_controller.extract_patches(mask, self.patch_size, self.patch_step)
        useful_patches = np.sum(mask_patches, axis=(1, 2, 3)) > self.util_patch_th
        return useful_patches

    def get_ids_by_index(self, index):
        return self.ids_loader[index]

    def get_iter_ids(self):
        return self.ids_loader.iter_ids
    
    def prepare_custom_patches(self, name, custom_img_sufix, data_type=np.float32, img_transform=None):
        self.custom_patches[name] = None
        if name not in self.custom_patches_metadata:
            self.custom_patches_metadata[name] = {"custom_img_sufix":custom_img_sufix, "data_type":data_type, "img_transform":img_transform}

        _cutom_patches_hunt3 = np.empty((0,) + self.patch_size, dtype=data_type)
        _cutom_patches_hunt4 = np.empty((0,) + self.patch_size, dtype=data_type)
        images_ids = self.ids_loader[self.current_index_batch_images]

        for i, id in enumerate(images_ids):
            img_hunt3 = nfc.load_nifti(self.ids_loader.get_custom_image_path_name(3, id, custom_img_sufix))[0].astype(data_type)
            img_hunt4 = nfc.load_nifti(self.ids_loader.get_custom_image_path_name(4, id, custom_img_sufix))[0].astype(data_type)

            if img_transform is not None:
                img_hunt3 = img_transform(img_hunt3)
                img_hunt4 = img_transform(img_hunt4)

             # extract patches
            _patches_hunt3 = patches_controller.extract_patches(img_hunt3, self.patch_size, self.patch_step)
            _patches_hunt4 = patches_controller.extract_patches(img_hunt4, self.patch_size, self.patch_step)
            
            # extract usefull patches
            _patches_hunt3 = _patches_hunt3[self.usefull_patches[id]]
            _patches_hunt4 = _patches_hunt4[self.usefull_patches[id]]

            # store usefull patches
            _cutom_patches_hunt3 = np.concatenate((_cutom_patches_hunt3, _patches_hunt3), axis=0)
            _cutom_patches_hunt4 = np.concatenate((_cutom_patches_hunt4, _patches_hunt4), axis=0)
            
        # shuffle usefull patches

        _cutom_patches_hunt3 = _cutom_patches_hunt3[self.patches_indexs]
        _cutom_patches_hunt4 = _cutom_patches_hunt4[self.patches_indexs]

        self.custom_patches[name] = (_cutom_patches_hunt3, _cutom_patches_hunt4)

    def get_iter_custom_patches(self, name):
        start, end = self.iter_patches_index
        custom_batch_hunt3 = self.custom_patches[name][0][start: end]
        custom_batch_hunt4 = self.custom_patches[name][1][start: end]

        custom_batch_hunt3 = np.expand_dims(custom_batch_hunt3, axis=1)
        custom_batch_hunt4 = np.expand_dims(custom_batch_hunt4, axis=1)
    
        if not self.channels_first:
            custom_batch_hunt3 = np.moveaxis(custom_batch_hunt3, 1, -1)
            custom_batch_hunt4 = np.moveaxis(custom_batch_hunt4, 1, -1)

        return custom_batch_hunt3, custom_batch_hunt4
    




def test_data_loader_patches():
    import project_images_loader as pil

    batch_size = 2
    image_size = (193,229,193)
    patch_size = (64,64,64)
    patch_step = (32,32,32)

    data_train = DataLoaderPatches(pil.INAME_PREPROCESSED, 
                                batch_size, 
                                image_size, 
                                patch_size, 
                                patch_step, 
                                images_onmemory=2, 
                                shuffle_patches=True, 
                                channels_first=False, 
                                background_th = 0.3,
                                    shuffle_images=False, 
                                    nb_patients=2, 
                                    shuffle_before_selecting=False, 
                                    server=False, 
                                    image_sufix_seg_gmm=pil.INAME_SEGMENTATION_3.format("{0}", "{1}", "nii.gz"))

    for i in data_train:
        print(data_train.iter_patches_index_ids)
        print(*data_train.get_iter_patient_gmm()[0][0])
        print(data_train.get_iter_patient_gmm().shape)
        break

# test_data_loader_patches()




