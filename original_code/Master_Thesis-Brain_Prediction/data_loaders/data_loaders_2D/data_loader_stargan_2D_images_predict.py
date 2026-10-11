import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(base_path)

import numpy as np
import utils.functions as fc

class DataLoaderImagesStarGAN2DPredict:

    HUNT3 = 0
    HUNT4 = 1
    def __init__(self, img_size, img_src, img_sty, mask_src, mask_sty, hunt_src, hunt_sty, channels_first=False, axis=0, dtype=np.float32):
        self.img_size = img_size
        self.img_src = img_src.astype(dtype)
        self.img_sty = img_sty.astype(dtype)
        self.mask_src = mask_src.astype(dtype)
        self.mask_sty = mask_sty.astype(dtype)
        self.hunt_src = hunt_src
        self.hunt_sty = hunt_sty
        self.axis = axis
        self.current_layer = 0
        self.channels_first = channels_first
        self.dtype = dtype  

  
    def __len__(self):
        return self.img_src.shape[self.axis]

    def __getitem__(self, layer):
        return self.__get_batch(layer)             

    def __iter__(self):
        for layer in self.__len__():
            yield self.__get_batch(layer)

    def __get_batch(self, layer):
        img_src_layer = self.__tranform_layer(fc.extract_layer(self.img_src, layer, self.axis))
        img_sty_layer = self.__tranform_layer(fc.extract_layer(self.img_sty, layer, self.axis))
        
        mask_src_layer = self.__tranform_layer(fc.extract_layer(self.mask_src, layer, self.axis))
        mask_sty_layer = self.__tranform_layer(fc.extract_layer(self.mask_sty, layer, self.axis))


        batch_images = [self.__organize_channels(img_src_layer), 
                        self.__organize_channels(img_sty_layer),
                        self.__organize_channels(mask_src_layer),
                        self.__organize_channels(mask_sty_layer),

                        ]
        

        classes = np.array([self.hunt_src, self.hunt_sty])
        classes = np.expand_dims(classes, axis=0)
        return batch_images, classes

    def __organize_channels(self, batch):
        batch = np.expand_dims(batch, axis=0)
        if self.channels_first:
            batch = np.moveaxis(batch, -1, 1)
        return batch
    
    def __tranform_layer(self, img):
        img = np.expand_dims(img, axis=-1)
        img = fc.pad_2D_img(img, self.img_size)
        return img
    
    # def get_class_from_hunt(self, hunt):
    #     if hunt == self.HUNT4:
    #         return 1
    #     else:
    #         return 0
   
    


def test_DataLoaderFileNamesStarGAN2D_2():
    import utils.functions as fc
    import utils.nifti_functions as nfc
    import utils.util as util
    import project_images_loader as pil
    import matplotlib.pyplot as plt
    import test_code.creator_2D_images.constants as cts

    ids=pil.get_matching_ids()

    id_src = ids[0]
    id_sty = ids[0]
    hunt_src = DataLoaderImagesStarGAN2DPredict.HUNT3
    hunt_sty =  DataLoaderImagesStarGAN2DPredict.HUNT4

    img_src = nfc.load_nifti(cts.get_registered_image_path_name(hunt_src+3, id_src))[0]
    bmask_src = nfc.load_nifti(cts.get_registered_bmask_path_name(hunt_src+3, id_src))[0]

    img_sty = nfc.load_nifti(cts.get_registered_image_path_name(hunt_sty+3, id_sty))[0]
    bmask_sty = nfc.load_nifti(cts.get_registered_bmask_path_name(hunt_sty+3, id_sty))[0]
    
    data_train = DataLoaderImagesStarGAN2DPredict(img_size=(256,256), 
                                           img_src=img_src,
                                           img_sty=img_sty, 
                                           mask_src=bmask_src,
                                           mask_sty=bmask_sty, 
                                           hunt_src=hunt_src, 
                                           hunt_sty=hunt_sty, 
                                           axis=0,
                                           channels_first=True)
    batch, classes = data_train[100]
    layer_src, layer_sty, layer_mask_src, layer_mask_sty = batch

    print(classes)
    print(util.get_basic_metrics_str(layer_src[0]), "\n", util.get_basic_metrics_str(layer_sty[0]), "\n", util.get_basic_metrics_str(layer_mask_src[0]))
    print(layer_src.shape, layer_sty.shape, layer_mask_src.shape, layer_mask_sty.shape)
    fc.imgshow_matplotlib(f"layer_src", layer_src[0,0])
    fc.imgshow_matplotlib(f"layer_sty", layer_sty[0,0])
    fc.imgshow_matplotlib(f"layer_mask_src", layer_mask_src[0,0])
    fc.imgshow_matplotlib(f"layer_mask_sty", layer_mask_sty[0,0])

    plt.show()

# test_DataLoaderFileNamesStarGAN2D_2()