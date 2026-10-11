import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(base_path)

import numpy as np
import random


class DataLoaderIdsStarGAN2D:

    HUNT4_FOLDER = "HUNT4"
    HUNT3_FOLDER = "HUNT3"
    # INAME = "{0}_{1}_PREP_{2}_TO_{3}.png"
    def __init__(self, path, image_sufix, ids, layers, batch_size, possible_table=None, shuffle=False, server=False, mode="train"):
        if len(ids) < 3:
            raise ValueError("Ids must have at least 3 elements")
        self.path = path
        self.image_sufix = image_sufix
        self.ids = ids
        self.layers = layers
        self.batch_size = batch_size
        self.possible_table = possible_table
        self.shuffle = shuffle
        self.server = server
        self.mode = mode

        self.iter_imgs_info = None

        self. index_list = self.__create_names_list()
        self.shuffle_indexs()
        
    def __create_names_list(self):
        index_matrix = np.indices((len(self.ids), len(self.ids), len(self.layers)), dtype=int).transpose(1, 2, 3, 0)
        index_list = index_matrix.reshape(-1, 3)

        if self.possible_table is not None:
            possible_index_matrix = []
            for index1, index2, layer in index_list:
                if self.possible_table[index1][index2] == 1:
                    possible_index_matrix.append([index1, index2, layer])

            index_list = np.array(possible_index_matrix)
        return index_list
    
    def shuffle_indexs(self):
        if self.shuffle:
            np.random.shuffle(self.index_list)
        
    def __len__(self):
        return int(np.ceil((self.index_list.shape[0])/self.batch_size))

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
                ret.append(self.__getitem__(i))
            self.iter_imgs_info = ret
            return self.iter_imgs_info
        
        else:
            if index < 0:
                index = self.__len__() + index
            self.iter_imgs_info = self.__get_batch(index)
            return self.iter_imgs_info         

    def __iter__(self):
        for i in range(self.__len__()):
            self.iter_imgs_info = self.__get_batch(i)
            yield self.iter_imgs_info

        self.shuffle_indexs()

    def __get_batch(self, batch_index):
        start = batch_index * self.batch_size
        end = min(start + self.batch_size, self.index_list.shape[0])

        _current_index = self.index_list[start:end]
        imgs_class = np.random.randint(0, 2, size=_current_index.shape[0])

        _batch_imgs_info = []
        for index, img_class_1 in zip(_current_index, imgs_class):
            id1, id2, layer = self.ids[index[0]], self.ids[index[1]], self.layers[index[2]]
            _imgs_info = [(id1, img_class_1, layer), (id2, 1-img_class_1, layer)]
            if self.mode == "train":
                id3 = self.ids[self.__get_possible_second_image_index(index[0], excluded_index=index[1])]
                _imgs_info.append((id3, 1-img_class_1, layer))
            _batch_imgs_info.append(_imgs_info)

        return _batch_imgs_info
    
    def get_images_path_name_from_info(self, batch_imgs_info):
        imgs_path_names = []
        for imgs_info in batch_imgs_info:
            img_path_names = []
            for id, img_class, layer in imgs_info:
                img_path_names.append(os.path.join(self.path, self.__get_class_folder(img_class), id, self.__create_image_name(id, img_class, layer)))
            imgs_path_names.append(img_path_names)
        return imgs_path_names
    
    def get_images_path_custom_name_from_info(self, batch_imgs_info, image_sufix):
        imgs_path_names = []
        for imgs_info in batch_imgs_info:
            img_path_names = []
            for id, img_class, layer in imgs_info:
                img_path_names.append(os.path.join(self.path, self.__get_class_folder(img_class), id, self.__create_custom_image_name(id, img_class, layer, image_sufix)))
            imgs_path_names.append(img_path_names)
        return imgs_path_names
    
    def get_images_classes_from_info(self, batch_imgs_info):
        return [imgs_info[0][1] for imgs_info in batch_imgs_info]
    
    
    def __get_class_folder(self, img_class):
        if img_class == 0:
            return self.HUNT3_FOLDER
        else:
            return self.HUNT4_FOLDER
        
    def __create_custom_image_name(self, id, img_class, layer, image_sufix):
        return image_sufix.format(id, img_class, layer)
    
    def __create_image_name(self, id, img_class, layer):
        return self.__create_custom_image_name(id, img_class, layer, self.image_sufix)
    

    def __get_possible_second_image_index(self, index, excluded_index):
        iterations = 0
        while True:
            iterations += 1
            possible_index = random.randint(0, len(self.ids)-1)
            if possible_index != index and possible_index != excluded_index:
                if self.possible_table is not None:
                    if self.possible_table[index][possible_index] == 1:
                        return possible_index
                else:
                    return possible_index
            if iterations > 100:
                raise ValueError("Infinite loop")


def test_DataLoaderFileNamesStarGAN2D():
    # ids=["a", "b", "c", "d"]
    # possible_table = [[1, 0, 1, 1],
    #                 [0, 1, 1, 1], 
    #                 [1, 1, 1, 1],
    #                 [1, 1, 1, 1]]

    ids=["a", "b", "c"]
    possible_table = [[1, 0, 1],
                    [0, 1, 1], 
                    [1, 1, 1]]
    layers=(1, 2)
    batch_size=1
    data_train = DataLoaderIdsStarGAN2D(path="",
                                        image_sufix="", 
                                        ids=ids, 
                                        layers=layers, 
                                        batch_size=batch_size, 
                                        possible_table=possible_table, 
                                        shuffle=False, 
                                        server=False, 
                                        mode="val")
    print(len(data_train))
    print(data_train[0:1])
    # print(data_train.iter_imgs_info)

    # print(data_train[0:5:2])
    # for i, (inames, classes) in enumerate(data_train):
    #     print(i, inames, classes)

# test_DataLoaderFileNamesStarGAN2D()


# def test_DataLoaderFileNamesStarGAN2D_2():
#     path = "C:/Users/Agustin/Master/Master/S4/code/test_code/stargan-v2-master/brain_2D_implementation/images_test/train"
#     ids=pil.get_matching_ids()[:3]

#     layers=[(34, 37), (37, 40)]
#     batch_size=2
#     data_train = DataLoaderFileNamesStarGAN2D(path, ids, layers, batch_size, possible_table=None, shuffle=False)
#     print(len(data_train))

#     for i, (inames, classes) in enumerate(data_train):
#         for iname in inames:
#             for sub_iname in iname:
#                 print(fc.verify_file(sub_iname))
#         # print(i, inames, classes)

# test_DataLoaderFileNamesStarGAN2D_2()

