import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../'))
sys.path.append(base_path)

import numpy as np
import utils.nifti_functions as nfc
import random
import utils.functions as fc
from data_loaders.data_loader_ids_clusters import DataLoaderIdsClusters

class DataLoaderBaselineClusters:

    HUNT3_FOLDER = "HUNT3"
    def __init__(self, image_sufix, clusters_csv_path, batch_size, image_size, channels_first=False, data_type=np.float32, 
                 return_clusters=False, onehot_encoder=True, train_fraction=0.8, shuffle_train=True, shuffle_val=False, exclude_ids=None, id_column=0, cluster_column=1, current_mode='train', server=False):
        self.batch_size = batch_size
        self.image_size = image_size
        self.channels_first = channels_first
        self.data_type = data_type

        self.ids_loader = DataLoaderIdsClusters(image_sufix=image_sufix, 
                                                data_path=clusters_csv_path,
                                                batch_size=batch_size, 
                                                return_clusters=return_clusters, 
                                                onehot_encoder=onehot_encoder, 
                                                train_fraction=train_fraction, 
                                                shuffle_train=shuffle_train, 
                                                shuffle_val=shuffle_val, 
                                                exclude_ids=exclude_ids, 
                                                id_column=id_column, 
                                                cluster_column=cluster_column, 
                                                current_mode=current_mode,
                                                server=server)


    def __len__(self):
        return len(self.ids_loader)

    def __getitem__(self, index):
        return self.__get_batch(self.ids_loader[index])
 
    def __iter__(self):
        for ids in self.ids_loader:
            yield self.__get_batch(ids)

    def __get_batch(self, ids):
        if self.ids_loader.return_clusters:
            ids, clusters = ids[0], ids[1]
        
        batch_size = (len(ids), 1, *self.image_size)
        batch_hunt3 = np.empty(batch_size, dtype=self.data_type)

        for i, id in enumerate(ids):
            img_hunt3, _ = nfc.load_nifti(self.ids_loader.get_image_path_name(3, id))
            batch_hunt3[i,0] = img_hunt3[:self.image_size[0], :self.image_size[1], :self.image_size[2]]
            
        if not self.channels_first:
            batch_hunt3 = np.moveaxis(batch_hunt3, 1, -1)

        if self.ids_loader.return_clusters:
            return batch_hunt3, clusters
        else:
            return batch_hunt3

    def get_ids_by_index(self, index):
        return self.ids_loader[index]

    def get_loop_ids(self):
        return self.ids_loader.loop_ids


    def change_mode(self, mode):
        self.ids_loader.change_mode(mode)

    def get_class_weights(self, alpha=1, mode='train'):
        return self.ids_loader.get_class_weights(alpha=alpha, mode=mode)

    def get_number_of_classes(self):
        return self.ids_loader.nb_clusters
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
    



