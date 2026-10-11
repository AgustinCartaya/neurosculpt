import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../'))
sys.path.append(base_path)

import numpy as np

import random
import utils.functions as fc
import project_images_loader as pil

class DataLoaderIds:

    HUNT3_FOLDER = "HUNT3"
    HUNT4_FOLDER = "HUNT4"
    def __init__(self, image_sufix, batch_size, shuffle=False, nb_patients=-1, shuffle_before_selecting=False, server=True, exclude_ids=None, start_from=0):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_sufix = image_sufix
        self.nb_patients = nb_patients
        self.shuffle_before_selecting = shuffle_before_selecting
        self.server = server
        self.exclude_ids = exclude_ids
        self.start_from = start_from

        # pil.SERVER = server
        self.__load_patient_ids()

        self.iter_ids = None


    def __load_patient_ids(self):
        self.patient_ids = self.__get_patients_ids_with_requested_image()
        
        # self.patient_ids = self.patient_ids[self.start_from:]
        # shuffle data
        # if self.shuffle and self.shuffle_before_selecting:
        #     self.__shuffle_ids()
        # select required number of images
     
        if self.exclude_ids is not None:
            self.patient_ids = [x for x in self.patient_ids if x not in self.exclude_ids]
        
        if ((self.start_from < 0) or (self.start_from >= len(self.patient_ids))):
            self.start_from = 0
        if ((self.nb_patients + self.start_from) > len(self.patient_ids)) or (self.nb_patients <= 0):
            self.nb_patients = len(self.patient_ids) - self.start_from

        self.patient_ids = self.patient_ids[self.start_from:self.start_from+self.nb_patients]

        if self.shuffle and not self.shuffle_before_selecting:
            self.shuffle_ids()

    def __len__(self):
        return int(np.ceil((self.nb_patients)/self.batch_size))


    def __getitem__(self, index):
        if isinstance(index, slice):
            _start = index.start if index.start is not None else 0
            _stop = index.stop if index.stop is not None else self.__len__()
            _step = index.step if index.step is not None else 1

            if _step < 0:
                _start, _stop = _stop - 1, _start - 1
                _step = abs(_step)

            ret = []
            for i in range(_start, _stop, _step):
                ret.extend(self.__getitem__(i))
            self.iter_ids = ret
            return ret
        
        else:
            if index < 0:
                index = self.__len__() + index

            start = index * self.batch_size
            end = min(start + self.batch_size, len(self.patient_ids))
            self.iter_ids = self.patient_ids[start:end]
            return self.iter_ids           


    def __iter__(self):
        for i in range(0, self.nb_patients, self.batch_size):
            self.iter_ids = self.patient_ids[i:min(i + self.batch_size, len(self.patient_ids))]
            yield self.iter_ids
        if self.shuffle:
            self.shuffle_ids()
        self.iter_ids = None


    def shuffle_ids(self):
        random.shuffle(self.patient_ids)

    def get_custom_image_path_name(self, hunt, patient_id, custom_img_sufix):
        return os.path.join(pil.get_patient_folder(hunt, patient_id, absolute=True), f"{custom_img_sufix.format(patient_id, hunt-3)}")
    
    def get_image_path_name(self, hunt, patient_id):
        return self.get_custom_image_path_name(hunt, patient_id, self.image_sufix)
    
    def get_image_bmask_path_name(self, hunt, patient_id):
        return os.path.join(pil.get_patient_folder(hunt, patient_id, absolute=True), f"{pil.INAME_BRAIN_MASK.format(patient_id, hunt-3, pil.REGISTRATION_TO_MNI)}")
    

    def __get_patients_ids_with_requested_image(self):
        patient_ids = pil.get_matching_ids()
        valid_ids = []
        for patient_id in patient_ids:
            if (fc.verify_file(self.get_image_path_name(3, patient_id)) and
                fc.verify_file(self.get_image_path_name(4, patient_id))):
                valid_ids.append(patient_id)
        
        return valid_ids

# input_shape = (192,224,192)
# batch_size = 10

# data_train = DataLoaderIds(image_sufix=pil.INAME_PREPROCESSED, 
#                            batch_size=batch_size, 
#                            shuffle=True, 
#                            nb_patients=20, 
#                            shuffle_before_selecting=False, 
#                            server=False)
# print(data_train[:2])

# for i in data_train:
#     print(i)

# print(data_train.patient_ids)
# data_train.patient_ids.sort()
# print(data_train.patient_ids)
# print(len(data_train.patient_ids))
    



