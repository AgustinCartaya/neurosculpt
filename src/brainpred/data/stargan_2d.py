"""2D (sagittal slice) data loaders and StarGAN v2 input fetchers for training and slice-wise prediction.

Training samples are triplets of PNG slices at the same position: a source slice of one patient in
domain ``c`` (0 = HUNT3, 1 = HUNT4) and two reference slices of two other patients in domain
``1 - c``. Slices are read from ``paths.get_images_folder_2D()/HUNT<3|4>/<id>/`` (see
:mod:`brainpred.preprocessing.slices_2d`).

The ``InputFetcher*`` classes follow the input fetcher of the official StarGAN v2 code
(https://github.com/clovaai/stargan-v2, CC BY-NC 4.0, see brainpred/methods/stargan_v2/LICENSE).
"""
import os
import random

import numpy as np
import torch
from munch import Munch
from PIL import Image
from torchvision import transforms

from brainpred.utils.arrays import extract_layer, pad_2D_img


# =============================================================================================
# Training: slice triplets from PNG files
# =============================================================================================
class DataLoaderIdsStarGAN2D:
    """Enumerates (source id, reference id, slice) combinations and builds batches of image infos."""

    HUNT4_FOLDER = "HUNT4"
    HUNT3_FOLDER = "HUNT3"

    def __init__(self, path, image_sufix, ids, layers, batch_size, possible_table=None, shuffle=False, mode="train"):
        """
        Args:
            path: root folder of the 2D slices.
            image_sufix: file name pattern ``{0}=id, {1}=class (hunt-3), {2}=slice``, e.g. ``paths.INAME2D_T1``.
            ids: patient ids.
            layers: slice indices.
            possible_table: optional ``len(ids) x len(ids)`` 0/1 table of allowed (source, reference) pairs.
            mode: ``train`` (two references per source) or ``val`` (one reference).
        """
        if len(ids) < 3:
            raise ValueError("Ids must have at least 3 elements")
        self.path = path
        self.image_sufix = image_sufix
        self.ids = ids
        self.layers = layers
        self.batch_size = batch_size
        self.possible_table = possible_table
        self.shuffle = shuffle
        self.mode = mode

        self.iter_imgs_info = None

        self.index_list = self.__create_names_list()
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
        return int(np.ceil((self.index_list.shape[0]) / self.batch_size))

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
            _imgs_info = [(id1, img_class_1, layer), (id2, 1 - img_class_1, layer)]
            if self.mode == "train":
                id3 = self.ids[self.__get_possible_second_image_index(index[0], excluded_index=index[1])]
                _imgs_info.append((id3, 1 - img_class_1, layer))
            _batch_imgs_info.append(_imgs_info)

        return _batch_imgs_info

    def get_images_path_name_from_info(self, batch_imgs_info):
        return self.get_images_path_custom_name_from_info(batch_imgs_info, self.image_sufix)

    def get_images_path_custom_name_from_info(self, batch_imgs_info, image_sufix):
        imgs_path_names = []
        for imgs_info in batch_imgs_info:
            img_path_names = []
            for id, img_class, layer in imgs_info:
                img_path_names.append(os.path.join(self.path, self.__get_class_folder(img_class), id,
                                                   image_sufix.format(id, img_class, layer)))
            imgs_path_names.append(img_path_names)
        return imgs_path_names

    def get_images_classes_from_info(self, batch_imgs_info):
        return [imgs_info[0][1] for imgs_info in batch_imgs_info]

    def __get_class_folder(self, img_class):
        return self.HUNT3_FOLDER if img_class == 0 else self.HUNT4_FOLDER

    def __get_possible_second_image_index(self, index, excluded_index):
        iterations = 0
        while True:
            iterations += 1
            possible_index = random.randint(0, len(self.ids) - 1)
            if possible_index != index and possible_index != excluded_index:
                if self.possible_table is not None:
                    if self.possible_table[index][possible_index] == 1:
                        return possible_index
                else:
                    return possible_index
            if iterations > 100:
                raise ValueError("Infinite loop")


class DataLoaderImagesStarGAN2D:
    """Loads the PNG slices of :class:`DataLoaderIdsStarGAN2D` batches, scaled to [0, 1] and zero-padded."""

    def __init__(self, img_size, n_channels, path, img_sufix, ids, layers, batch_size, possible_table=None, shuffle=False,
                 channels_first=False, mode="train", dtype=np.float32):
        self.img_size = img_size
        self.n_channels = n_channels
        self.channels_first = channels_first
        self.dtype = dtype

        self.dloader_ids = DataLoaderIdsStarGAN2D(path, img_sufix, ids, layers, batch_size, possible_table=possible_table,
                                                  shuffle=shuffle, mode=mode)

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

        batch_images = [self.__organize_channels(batch_src), self.__organize_channels(batch_sty1)]

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

        if normalize:
            img = img / 255.0
        if pad:
            img = pad_2D_img(img, self.img_size)
        return img

    def shuffle_data(self):
        self.dloader_ids.shuffle_indexs()

    def get_mode(self):
        return self.dloader_ids.mode

    def get_batch_imgs_info(self):
        return self.dloader_ids.iter_imgs_info

    def get_iter_custom_image(self, img_sufix, dtype=np.float32, normalize=True, pad=True):
        """Other images (e.g. brain masks, ``img_sufix=paths.INAME2D_BMASK``) of the last fetched batch."""
        imgs_info = self.dloader_ids.iter_imgs_info

        batch_src = np.empty((len(imgs_info), *self.img_size, self.n_channels), dtype=dtype)
        batch_sty1 = np.empty((len(imgs_info), *self.img_size, self.n_channels), dtype=dtype)

        imgs_path_names = self.dloader_ids.get_images_path_custom_name_from_info(imgs_info, img_sufix)
        for i, image_name in enumerate(imgs_path_names):
            batch_src[i] = self.__open_img(image_name[0], normalize=normalize, pad=pad)
            batch_sty1[i] = self.__open_img(image_name[1], normalize=normalize, pad=pad)

        batch_images = [self.__organize_channels(batch_src), self.__organize_channels(batch_sty1)]

        if self.dloader_ids.mode == "train":
            batch_sty2 = np.empty((len(imgs_info), *self.img_size, self.n_channels), dtype=dtype)
            for i, image_name in enumerate(imgs_path_names):
                batch_sty2[i] = self.__open_img(image_name[2])
            batch_images.append(self.__organize_channels(batch_sty2))

        return batch_images


class InputFetcher:
    """Endless iterator of StarGAN v2 training / validation inputs (images normalised to [-1, 1])."""

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
        x = torch.tensor(x)
        if normalize:
            x = transforms.Normalize(self.mean, self.std)(x)
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
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref, x_ref=x_ref, x_ref2=x_ref2, z_trg=z_trg, z_trg2=z_trg2)
        else:
            (x, x_ref), (y, y_ref) = self._fetch()
            inputs = Munch(x_src=x, y_src=y, x_ref=x_ref, y_ref=y_ref)

        return Munch({k: v.to(self.device) for k, v in inputs.items()})

    def get_iter_custom_images(self, img_sufix, dtype=np.float32, normalize=True, standarize=True, pad=True, only_src=False):
        """Other images (e.g. brain masks) of the last fetched batch as tensors."""
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

        return Munch({k: v.to(self.device) for k, v in inputs.items()})


# =============================================================================================
# Prediction: slices of in-memory 3D volumes
# =============================================================================================
class DataLoaderImagesStarGAN2DPredict:
    """Yields, slice by slice along ``axis``, the source / style slices and their brain masks (batch of 1)."""

    HUNT3 = 0
    HUNT4 = 1

    def __init__(self, img_size, img_src, img_sty, mask_src, mask_sty, hunt_src, hunt_sty, channels_first=False, axis=0,
                 dtype=np.float32):
        self.img_size = img_size
        self.img_src = img_src.astype(dtype)
        self.img_sty = img_sty.astype(dtype)
        self.mask_src = mask_src.astype(dtype)
        self.mask_sty = mask_sty.astype(dtype)
        self.hunt_src = hunt_src
        self.hunt_sty = hunt_sty
        self.axis = axis
        self.channels_first = channels_first
        self.dtype = dtype

    def __len__(self):
        return self.img_src.shape[self.axis]

    def __getitem__(self, layer):
        return self.__get_batch(layer)

    def __iter__(self):
        for layer in range(len(self)):
            yield self.__get_batch(layer)

    def __get_batch(self, layer):
        batch_images = [self.__organize_channels(self.__tranform_layer(extract_layer(volume, layer, self.axis)))
                        for volume in (self.img_src, self.img_sty, self.mask_src, self.mask_sty)]
        classes = np.expand_dims(np.array([self.hunt_src, self.hunt_sty]), axis=0)
        return batch_images, classes

    def __organize_channels(self, batch):
        batch = np.expand_dims(batch, axis=0)
        if self.channels_first:
            batch = np.moveaxis(batch, -1, 1)
        return batch

    def __tranform_layer(self, img):
        img = np.expand_dims(img, axis=-1)
        return pad_2D_img(img, self.img_size)


class InputFetcherPredict:
    """Iterates once over the slices of a :class:`DataLoaderImagesStarGAN2DPredict` (images in [-1, 1], masks as-is)."""

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
        y_src = torch.tensor(batch_y[:, 0], dtype=torch.int)
        y_ref = torch.tensor(batch_y[:, 1], dtype=torch.int)
        return (*batch_tensor_imgs, *batch_tensor_mask), (y_src, y_ref)

    def __iter__(self):
        return self

    def __len__(self):
        return self.max_i

    def __next__(self):
        self.current_i += 1
        if self.current_i >= self.max_i:
            raise StopIteration

        (x, x_ref, mask, mask_ref), (y, y_ref) = self._fetch()
        inputs = Munch(x_src=x, y_src=y, x_ref=x_ref, y_ref=y_ref, mask=mask, mask_ref=mask_ref)
        return Munch({k: v.to(self.device) for k, v in inputs.items()})
