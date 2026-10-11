"""Batches of whole (cropped) HUNT3 / HUNT4 volume pairs."""
import numpy as np

from brainpred.data.ids import DataLoaderIds
from brainpred.utils import io


class DataLoaderComplete:
    """Yield ``(batch_hunt3, batch_hunt4)`` of shape ``(batch, 1, *image_size)`` (or channels-last).

    Volumes are cropped to ``image_size`` starting at ``img_size_offset``. See :class:`DataLoaderIds`
    for the patient-selection arguments.
    """

    def __init__(self, image_sufix, batch_size, image_size, channels_first=False, shuffle=False, nb_patients=-1,
                 data_type=np.float32, exclude_ids=None, start_from=0, img_size_offset=(0, 0, 0), patient_ids=None):
        self.batch_size = batch_size
        self.image_size = image_size
        self.channels_first = channels_first
        self.data_type = data_type
        self.img_size_offset = img_size_offset
        self.ids_loader = DataLoaderIds(image_sufix=image_sufix,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        nb_patients=nb_patients,
                                        exclude_ids=exclude_ids,
                                        start_from=start_from,
                                        patient_ids=patient_ids)

    def __len__(self):
        return len(self.ids_loader)

    def __getitem__(self, index):
        return self.__get_batch(self.ids_loader[index])

    def __iter__(self):
        for ids in self.ids_loader:
            yield self.__get_batch(ids)

    def _crop(self, img):
        x0, y0, z0 = self.img_size_offset
        return img[x0:x0 + self.image_size[0], y0:y0 + self.image_size[1], z0:z0 + self.image_size[2]]

    def _load_pair_batch(self, ids, img_sufix, data_type, transform=None):
        batch_shape = (len(ids), 1, *self.image_size)
        batch_hunt3 = np.empty(batch_shape, dtype=data_type)
        batch_hunt4 = np.empty(batch_shape, dtype=data_type)

        for i, id in enumerate(ids):
            img_hunt3 = io.load_nifti(self.ids_loader.get_custom_image_path_name(3, id, img_sufix))[0].astype(data_type)
            img_hunt4 = io.load_nifti(self.ids_loader.get_custom_image_path_name(4, id, img_sufix))[0].astype(data_type)
            if transform is not None:
                img_hunt3, img_hunt4 = transform(img_hunt3), transform(img_hunt4)
            batch_hunt3[i, 0] = self._crop(img_hunt3)
            batch_hunt4[i, 0] = self._crop(img_hunt4)

        if not self.channels_first:
            batch_hunt3 = np.moveaxis(batch_hunt3, 1, -1)
            batch_hunt4 = np.moveaxis(batch_hunt4, 1, -1)
        return batch_hunt3, batch_hunt4

    def __get_batch(self, ids):
        return self._load_pair_batch(ids, self.ids_loader.image_sufix, self.data_type)

    def get_ids_by_index(self, index):
        return self.ids_loader[index]

    def get_iter_ids(self):
        return self.ids_loader.iter_ids

    def get_iter_custom_images(self, custom_img_sufix, data_type=np.float32, transform=None):
        """Load another image type (e.g. a segmentation) for the IDs of the current batch."""
        return self._load_pair_batch(self.get_iter_ids(), custom_img_sufix, data_type, transform)
