import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../'))
sys.path.append(base_path)

import numpy as np
import utils.nifti_functions as nfc
import random
import utils.functions as fc
from data_loaders.data_loader_ids import DataLoaderIds

class DataLoaderBaseline:

    HUNT3_FOLDER = "HUNT3"
    def __init__(self, image_sufix, batch_size, image_size, channels_first=False, shuffle=False, nb_patients=-1, shuffle_before_selecting=False, server=True, data_type=np.float32, exclude_ids=None, start_from=0):
        self.batch_size = batch_size
        self.image_size = image_size
        self.channels_first = channels_first
        self.data_type = data_type

        self.ids_loader = DataLoaderIds(image_sufix=image_sufix, 
                                        batch_size=batch_size, 
                                        shuffle=shuffle, 
                                        nb_patients=nb_patients, 
                                        shuffle_before_selecting=shuffle_before_selecting, 
                                        server=server,
                                        exclude_ids=exclude_ids,
                                        start_from=start_from)

    def __len__(self):
        return len(self.ids_loader)

    def __getitem__(self, index):
        return self.__get_batch(self.ids_loader[index])
 
    def __iter__(self):
        for ids in self.ids_loader:
            yield self.__get_batch(ids)

    def __get_batch(self, ids):
        batch_size = (len(ids), 1, *self.image_size)
        batch_hunt3 = np.empty(batch_size, dtype=self.data_type)

        for i, id in enumerate(ids):
            img_hunt3, _ = nfc.load_nifti(self.ids_loader.get_image_path_name(3, id))
            batch_hunt3[i,0] = img_hunt3[:self.image_size[0], :self.image_size[1], :self.image_size[2]]
            
        if not self.channels_first:
            batch_hunt3 = np.moveaxis(batch_hunt3, 1, -1)

        return batch_hunt3

    def get_ids_by_index(self, index):
        return self.ids_loader[index]

    def get_current_ids(self):
        return self.ids_loader.current_ids




# import project_images_loader as pil

# input_shape = (192,224,192)
# batch_size = 10

# data_train = DataLoaderComplete(
#                         pil.INAME_PREPROCESSED,
#                         batch_size, 
#                         input_shape, 
#                         channels_first=True, 
#                         shuffle=False, 
#                         nb_patients=-1, 
#                         shuffle_before_selecting=False, 
#                         server=False)

# print(data_train.get_ids_by_index(5)[0])
# fc.imgshow_3D("", data_train[5:8][0][0,0])
# print(data_train[5:8][0].shape)

# print(data_train.patient_ids)
# data_train.patient_ids.sort()
# print(data_train.patient_ids)
# print(len(data_train.patient_ids))
    



