import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../'))
sys.path.append(base_path)

import numpy as np

import torch
from munch import Munch
from torchvision import transforms

class InputFetcher:
    def __init__(self, loader, latent_dim=16):
        self.loader = loader
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.current_i = 0
        self.max_i = len(loader)

        if loader.n_channels == 3:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        elif loader.n_channels == 1:
            self.mean = [0.5]
            self.std = [0.5]

    def __apply_transform(self, x, normalize=True):
        # print("x shape:", x.shape)
        x = torch.tensor(x)
        if normalize:
            x = transforms.Normalize(self.mean, self.std)(x)
        # x = torch.stack([transforms.functional.to_tensor(img) for img in x])
        # x = torch.stack([transforms.functional.normalize(img, self.mean, self.std) for img in x])
        return x
    

    def _fetch(self):
        batch_images, y_src = self.loader[self.current_i]
        batch_tensor = [self.__apply_transform(batch) for batch in batch_images]
        y_src = torch.tensor(y_src, dtype=torch.int)
        y_ref = 1 - y_src

        return batch_tensor, (y_src, y_ref)  


    def __iter__(self):
        return self
    
    def __next__(self):
        if self.loader.get_mode() in ['train', 'val']:
            self.current_i = (self.current_i + 1) % self.max_i 
        else:
            raise NotImplementedError
        
        if self.loader.get_mode() == 'train':
            (x, x_ref, x_ref2), (y, y_ref) = self._fetch()

            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref2,
                           z_trg=z_trg, z_trg2=z_trg2)
            
        elif self.loader.get_mode() == 'val':
            (x, x_ref), (y, y_ref) = self._fetch()
            inputs = Munch(x_src=x, y_src=y,
                           x_ref=x_ref, y_ref=y_ref)

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})
    

    def get_iter_custom_images(self, img_sufix, dtype=np.float32, normalize=True, standarize=True, pad=True, only_src=False):
        batch_images = self.loader.get_iter_custom_image(img_sufix, dtype=dtype, normalize=normalize, pad=pad)
        batch_tensor = [self.__apply_transform(batch, standarize) for batch in batch_images]
        
        if only_src:
            inputs = Munch(x_cust_src=batch_tensor[0])

        elif self.loader.get_mode() == 'train':
            x, x_ref1, x_ref2 = batch_tensor
            inputs = Munch(x_cust_src=x, x_cust_ref=x_ref1, x_cust_ref2=x_ref2)
        elif self.loader.get_mode() == 'val':
            x, x_ref = batch_tensor
            inputs = Munch(x_cust_src=x, x_cust_ref=x_ref)
        
        return Munch({k: v.to(self.device)
                    for k, v in inputs.items()})

def test():
    from data_loaders.data_loaders_2D.data_loader_stargan_2D_images import DataLoaderImagesStarGAN2D
    import utils.functions as fc
    import utils.util as util
    import project_images_loader as pil
    import matplotlib.pyplot as plt

    path = pil.get_images_folder_2D(absolute=True)
    ids=pil.get_matching_ids()[:100]

    layers=[layer for layer in range(pil.LAYER_START_2D, pil.LAYER_END_2D)]
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
                                           shuffle=True, 
                                           server=False,
                                           channels_first=True,
                                           mode="val",
                                           dtype=np.float32)
    
    data_train_fetcher = InputFetcher(data_train, 16)
    inputs = next(data_train_fetcher)
    x_real, y_org = inputs.x_src, inputs.y_src
    x_ref, y_trg = inputs.x_ref,  inputs.y_ref

    inputs_cust = data_train_fetcher.get_iter_custom_images(pil.INAME2D_BMASK, dtype=np.float32, normalize=True, standarize=False, pad=True, only_src=False)
    x_cust_src, x_cust_ref = inputs_cust.x_cust_src,  inputs_cust.x_cust_ref

    x_real = x_real.cpu().numpy()
    y_org = y_org.cpu().numpy()
    x_ref = x_ref.cpu().numpy()
    y_trg = y_trg.cpu().numpy()

    x_cust_src = x_cust_src.cpu().numpy()
    x_cust_ref = x_cust_ref.cpu().numpy()

    print("Original shapes:", x_real.shape, y_org.shape)
    print("References shapes:", x_ref.shape, y_trg.shape)

    print("Custom shapes:", x_cust_ref.shape, x_cust_src.shape)
    print("Custom vals:", np.unique(x_cust_ref), np.unique(x_cust_ref))

    print(util.get_basic_metrics_str(x_real[0]))
    print(util.get_basic_metrics_str(x_ref[0]))

    fc.imgshow_matplotlib(f"x_real {y_org[0]}", x_real[0].transpose(1,2,0))
    fc.imgshow_matplotlib(f"x_ref {y_trg[0]}", x_ref[0].transpose(1,2,0))

    fc.imgshow_matplotlib(f"x_cust_src {y_trg[0]}", x_cust_src[0].transpose(1,2,0))
    fc.imgshow_matplotlib(f"x_cust_ref {y_trg[0]}", x_cust_ref[0].transpose(1,2,0))
    plt.show()


# test()