import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(base_path)

import numpy as np

import torch
from munch import Munch
from torchvision import transforms

class InputFetcherPredict:
    def __init__(self, loader_eval):
        self.loader = loader_eval
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.current_i = -1
        self.max_i = len(loader_eval)
        self.mean = [0.5]
        self.std = [0.5]

    def __apply_transform(self, x, normalize=True):
        x = torch.tensor(x)
        if normalize:
            x = transforms.Normalize(self.mean, self.std)(x)
        return x
    

    def _fetch(self):
        batch_images, batch_y = self.loader[self.current_i]
        batch_tensor_imgs = [self.__apply_transform(batch) for batch in batch_images[:2]]
        batch_tensor_mask = [self.__apply_transform(batch, normalize=False) for batch in batch_images[2:]]
        y_src = batch_y[:,0]
        y_ref = batch_y[:,1]
        y_src = torch.tensor(y_src, dtype=torch.int)
        y_ref = torch.tensor(y_ref, dtype=torch.int)
        return (*batch_tensor_imgs, *batch_tensor_mask), (y_src, y_ref)  

    def __iter__(self):
        return self
    
    def __next__(self): 
        self.current_i += 1
        if self.current_i >= self.max_i:
            raise StopIteration
     
        (x, x_ref, mask, mask_ref), (y, y_ref) = self._fetch()
        inputs = Munch(x_src=x, y_src=y,
                        x_ref=x_ref, y_ref=y_ref,
                        mask=mask, mask_ref=mask_ref)
        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})
    


   



def test():
    import utils.functions as fc
    import utils.nifti_functions as nfc
    import utils.util as util
    import project_images_loader as pil
    import matplotlib.pyplot as plt
    import test_code.creator_2D_images.constants as cts
    from data_loaders.data_loaders_2D.data_loader_stargan_2D_images_predict import DataLoaderImagesStarGAN2DPredict

    ids=pil.get_matching_ids()

    id_src = ids[1]
    id_sty = ids[1]
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
    # batch, classes = data_train[100]
    # layer_src, layer_sty, layer_mask_src, layer_mask_sty = batch


    data_train_fetcher = InputFetcherPredict(data_train)
    for i in range(100):
        inputs = next(data_train_fetcher)
    x_real, y_org, mask_real = inputs.x_src, inputs.y_src, inputs.mask
    x_ref, y_trg, mask_ref = inputs.x_ref,  inputs.y_ref, inputs.mask_ref

    x_real = x_real.cpu().numpy()
    y_org = y_org.cpu().numpy()
    mask_real = mask_real.cpu().numpy()

    x_ref = x_ref.cpu().numpy()
    y_trg = y_trg.cpu().numpy()
    mask_ref = mask_ref.cpu().numpy()

    print("Original shapes:", x_real.shape, y_org.shape)
    print("References shapes:", x_ref.shape, y_trg.shape)

    print("Custom shapes:", mask_real.shape, mask_ref.shape)
    print("Custom vals:", np.unique(mask_real), np.unique(mask_ref))

    print(util.get_basic_metrics_str(x_real[0]))
    print(util.get_basic_metrics_str(x_ref[0]))

    fc.imgshow_matplotlib(f"x_real {y_org[0]}", x_real[0].transpose(1,2,0))
    fc.imgshow_matplotlib(f"x_ref {y_trg[0]}", x_ref[0].transpose(1,2,0))

    fc.imgshow_matplotlib(f"x_cust_src {y_trg[0]}", mask_real[0].transpose(1,2,0))
    fc.imgshow_matplotlib(f"x_cust_ref {y_trg[0]}", mask_ref[0].transpose(1,2,0))
    plt.show()


# test()