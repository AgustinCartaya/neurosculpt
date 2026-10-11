import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(base_path)


import numpy as np
import random
from data_loaders.data_loaders_2D.data_loader_stargan_2D_ids import DataLoaderIdsStarGAN2D
from PIL import Image
import utils.functions as fc

class DataLoaderImagesStarGAN2D:


    def __init__(self, img_size, n_channels, path, img_sufix, ids, layers, batch_size, possible_table=None, shuffle=False, server=False, channels_first=False, mode="train", dtype=np.float32):
        self.img_size = img_size
        self.n_channels = n_channels
        self.channels_first = channels_first
        self.dtype = dtype

        self.dloader_ids = DataLoaderIdsStarGAN2D(path,
                                                    img_sufix,
                                                    ids, 
                                                    layers, 
                                                    batch_size, 
                                                    possible_table=possible_table, 
                                                    shuffle=shuffle,
                                                    server=server,
                                                    mode=mode)

            

    def __len__(self):
        return len(self.dloader_ids)

    def __getitem__(self, index):
        return self.__get_batch(self.dloader_ids[index])             

    def __iter__(self):
        for ids in self.dloader_ids:
            yield self.__get_batch(ids)

    def __get_batch(self, ids):
        batch_src = np.empty((len(ids), *self.img_size, self.n_channels), dtype=self.dtype)
        batch_sty1 = np.empty((len(ids), *self.img_size, self.n_channels), dtype=self.dtype)
        
        imgs_path_names = self.dloader_ids.get_images_path_name_from_info(ids)

        for i, image_name in enumerate(imgs_path_names):              
            batch_src[i] = self.__open_img(image_name[0])
            batch_sty1[i] = self.__open_img(image_name[1])

        batch_images = [self.__organize_channels(batch_src), 
                        self.__organize_channels(batch_sty1)]
        
        if self.dloader_ids.mode == "train":
            batch_sty2 = np.empty((len(ids), *self.img_size, self.n_channels), dtype=self.dtype)
            for i, image_name in enumerate(imgs_path_names):              
                batch_sty2[i] = self.__open_img(image_name[2])
            batch_images.append(self.__organize_channels(batch_sty2))

        classes = self.dloader_ids.get_images_classes_from_info(ids)
        return batch_images, classes

    def __organize_channels(self, batch):
        if self.channels_first:
            batch = np.moveaxis(batch, -1, 1)
        return batch
    
    def __open_img(self, img_name, normalize=True, pad=True):
        if self.n_channels == 1:
            img = np.array(Image.open(img_name), dtype=self.dtype)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
        else:
            img = np.array(Image.open(img_name).convert('RGB'), dtype=self.dtype)

        # normalize
        if normalize:
            img = img/255.0

        # pad
        if pad:
            img = fc.pad_2D_img(img, self.img_size)
        return img
    
    def shuffle_data(self):
        self.dloader_ids.shuffle_indexs()

    def get_mode(self):
        return self.dloader_ids.mode
    
    def get_batch_imgs_info(self):
        return self.dloader_ids.iter_imgs_info
    
    def get_iter_custom_image(self, img_sufix, dtype=np.float32, normalize=True, pad=True):
        imgs_info = self.dloader_ids.iter_imgs_info

        batch_src = np.empty((len(imgs_info), *self.img_size, self.n_channels), dtype=dtype)
        batch_sty1 = np.empty((len(imgs_info), *self.img_size, self.n_channels), dtype=dtype)
        
        imgs_path_names = self.dloader_ids.get_images_path_custom_name_from_info(imgs_info, img_sufix)

        for i, image_name in enumerate(imgs_path_names):              
            batch_src[i] = self.__open_img(image_name[0], normalize=normalize, pad=pad)
            batch_sty1[i] = self.__open_img(image_name[1], normalize=normalize, pad=pad)

        batch_images = [self.__organize_channels(batch_src), 
                        self.__organize_channels(batch_sty1)]
        
        if self.dloader_ids.mode == "train":
            batch_sty2 = np.empty((len(imgs_info), *self.img_size, self.n_channels), dtype=dtype)
            for i, image_name in enumerate(imgs_path_names):              
                batch_sty2[i] = self.__open_img(image_name[2])
            batch_images.append(self.__organize_channels(batch_sty2))

        return batch_images

        
    


def test_DataLoaderFileNamesStarGAN2D_2():
    import utils.functions as fc
    import utils.util as util
    import project_images_loader as pil
    import matplotlib.pyplot as plt

    path = pil.get_images_folder_2D(absolute=True)
    ids=pil.get_matching_ids()[:100]

    layers=[34, 40]
    batch_size=1
    img_sufix = pil.INAME2D_T1
    data_train = DataLoaderImagesStarGAN2D(img_size=(256,256), 
                                           n_channels=1,
                                           path=path, 
                                           img_sufix=img_sufix,
                                           ids=ids, 
                                           layers=layers, 
                                           batch_size=batch_size, 
                                           possible_table=None, 
                                           shuffle=False, 
                                           server=False,
                                           channels_first=True,
                                           mode="train",
                                           dtype=np.float32)
    batch, classes = data_train[0]
    img1, img2, img3 = batch

    batch_cust = data_train.get_iter_custom_image(pil.INAME2D_SEG_3, dtype=np.float32, normalize=False, pad=True)
    img1_cust, img2_cust, img3_cust = batch_cust


    batch_imgs_info = data_train.get_batch_imgs_info()
    print(batch_imgs_info)
    print(classes)
    print(img1.shape, img2.shape, img3.shape)
    print(util.get_basic_metrics_str(img1[0]), "\n", util.get_basic_metrics_str(img2[0]), "\n", util.get_basic_metrics_str(img3[0]))

    fc.imgshow_matplotlib(f"{batch_imgs_info[0][0]}", img1[0,0])
    fc.imgshow_matplotlib(f"{batch_imgs_info[0][1]}", img2[0,0])
    fc.imgshow_matplotlib(f"{batch_imgs_info[0][2]}", img3[0,0])

    fc.imgshow_matplotlib(f"{batch_imgs_info[0][0]}", img1_cust[0,0])
    fc.imgshow_matplotlib(f"{batch_imgs_info[0][1]}", img2_cust[0,0])
    fc.imgshow_matplotlib(f"{batch_imgs_info[0][2]}", img3_cust[0,0])
    plt.show()

# test_DataLoaderFileNamesStarGAN2D_2()