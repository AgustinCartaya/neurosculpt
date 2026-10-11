import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../'))
sys.path.append(base_path)

import numpy as np
import pandas as pd

import random
import utils.functions as fc
import project_images_loader as pil
from sklearn.preprocessing import OneHotEncoder

class DataLoaderIdsClusters:
    def __init__(self, image_sufix, data_path, batch_size, return_clusters=False, onehot_encoder=True, train_fraction=0.8, shuffle_train=True, shuffle_val=False, exclude_ids=None, id_column=0, cluster_column=1, current_mode='train', server=False):
        self.image_sufix = image_sufix
        self.data_path = data_path
        self.batch_size = batch_size
        self.return_clusters = return_clusters
        self.onehot_encoder = onehot_encoder
        self.train_fraction = train_fraction
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.current_mode = current_mode
        self.exclude_ids = [int(x) for x in exclude_ids] if exclude_ids is not None else []
        self.id_column = id_column
        self.cluster_column = cluster_column
        self.server=server

        # pil.SERVER = server

        self.patient_ids, self.clusters = self.__load_data()
        self.clusters_onehot = self.__obtain_onehot_clusters()

        self.cluster_count = np.unique(self.clusters, return_counts=True)[1]
        self.nb_clusters = len(self.cluster_count)

        if train_fraction < 1.0:
            self.patient_indexs_train, self.patient_indexs_val = self.__divide_train_test_ids()
        else:
            self.patient_indexs_train, self.patient_indexs_val = [i for i in range(len(self.patient_ids))], []

        self.current_patient_indexs = None
        self.current_nb_patients = 0
        self.change_mode(current_mode)
        self.__shuffle_current_indexs()


        self.loop_ids = None
        self.loop_cluster = None


    def __obtain_onehot_clusters(self):
        encoder = OneHotEncoder()
        return encoder.fit_transform(self.clusters.reshape(-1, 1)).toarray()
        
    def __load_data(self):
        _df = pd.read_csv(f'{self.data_path}')
        _df = _df[~_df.iloc[:, self.id_column].isin(self.exclude_ids)]    

        _patient_ids = _df.iloc[:, self.id_column].values.astype(int)
        _clusters = _df.iloc[:, self.cluster_column].values
        return _patient_ids, _clusters
    

    def __divide_train_test_ids(self):
        _nb_patients_per_cluster_train = np.ceil(self.cluster_count * self.train_fraction).astype(int)
        for i in range(self.nb_clusters):
            if _nb_patients_per_cluster_train[i] == self.cluster_count[i] and self.cluster_count[i] > 1:
                _nb_patients_per_cluster_train[i] -= 1

        _train_indexs = []
        _val_indexs = []

        
        self.nb_patients_per_cluster_train = _nb_patients_per_cluster_train
        self.nb_patients_per_cluster_val = self.cluster_count - _nb_patients_per_cluster_train

        for i in range(self.nb_clusters):
            _cluster_index = np.where(self.clusters.astype(int) == i)[0]
            _train_indexs.extend(_cluster_index[:_nb_patients_per_cluster_train[i]].tolist())
            _val_indexs.extend(_cluster_index[_nb_patients_per_cluster_train[i]:].tolist())
    
        print(" ------------------- Data Loader ------------------- ")
        print(f"Total: {len(self.patient_ids)}, Per clusters: {self.cluster_count}")
        print(f"Nb images Train: {len(_train_indexs)}, Val: {len(_val_indexs)}")
        print(f"Nb images per cluster Train: {_nb_patients_per_cluster_train}, Val: {(self.cluster_count-_nb_patients_per_cluster_train)}")
        return _train_indexs, _val_indexs
    
    def __shuffle_current_indexs(self):
        if (self.shuffle_train and self.current_mode == 'train') or (self.shuffle_val and self.current_mode == 'val'):
            random.shuffle(self.current_patient_indexs)
 

    def __len__(self):
        return int(np.ceil(self.current_nb_patients / self.batch_size))


    def __get_batch(self, indexs):
        self.loop_ids = ["{0:05}".format(id) for id in self.patient_ids[indexs]]
        if self.return_clusters:
            if self.onehot_encoder:
                self.loop_clusters = self.clusters_onehot[indexs]
            else:
                self.loop_clusters = self.clusters[indexs]
            return self.loop_ids, self.loop_clusters
        else:
            return self.loop_ids

    def __iter__(self):
        for i in range(0, self.current_nb_patients, self.batch_size):
            _current_indexs = self.current_patient_indexs[i:min(i + self.batch_size, self.current_nb_patients)]
            yield self.__get_batch(_current_indexs)

        self.__shuffle_current_indexs()


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
            return ret
        else:
            if index < 0:
                index = self.__len__() + index

            start = index * self.batch_size
            end = min(start + self.batch_size, self.current_nb_patients)
            _current_indexs = self.current_patient_indexs[start:end]
            return self.__get_batch(_current_indexs)


    def change_mode(self, mode):
        if mode == 'train':
            self.current_patient_indexs = self.patient_indexs_train
        elif mode == 'val':
            self.current_patient_indexs = self.patient_indexs_val
        else:
            raise ValueError(f"Mode {mode} is not valid. Use 'train' or 'val'.")
        
        self.current_nb_patients = len(self.current_patient_indexs)
        self.current_mode = mode

    def get_class_weights(self, alpha=1, mode='train'):
        if mode == 'train':
            _counts = self.nb_patients_per_cluster_train
        elif mode == 'val':
            _counts = self.nb_patients_per_cluster_val
        else:
            raise ValueError(f"Mode {mode} is not valid. Use 'train' or 'val'.")
        return alpha/_counts
    
    def get_clusters(self, ids):
        ids = [int(x) for x in ids]
        index = np.where(np.isin(self.patient_ids, ids))[0]
        if self.onehot_encoder:
            return self.clusters_onehot[index]
        return self.clusters[index]    
    
    def get_image_path_name(self, hunt, patient_id):
        return os.path.join(pil.get_patient_folder(hunt, patient_id, absolute=True), f"{self.image_sufix.format(patient_id, hunt-3)}")
    




# data_loader = DataLoaderIdsClusters(pil.INAME_PREPROCESSED, 
#                                     f"{current_path}/clusters/C16_T96_vol_diff_perc.csv", 
#                                     batch_size=10, 
#                                     return_clusters=True, 
#                                     exclude_ids=None, 
#                                     id_column=0, 
#                                     cluster_column=1, 
#                                     onehot_encoder=False, 
#                                     train_fraction=0.8, 
#                                     shuffle_train=False, 
#                                     shuffle_val=False, 
#                                     current_mode='train')

# print(data_loader.patient_ids)
# print(data_loader.clusters)
# print(data_loader.current_nb_patients)
# print(data_loader.current_patient_indexs)


# print(data_loader[0])

# data_loader.change_mode('val')
# print(data_loader[0])

# print(data_loader.nb_patients_per_cluster_train)
# print(data_loader.nb_patients_per_cluster_val)

# print(data_loader.get_class_weights())
# print(data_loader.get_class_weights(mode='val'))

# for x, y in data_loader:
#     print(x)
#     print(y)
    
# data_loader.change_mode('val')
# print("val")
# for x, y in data_loader:
#     print(x)
#     print(y)
