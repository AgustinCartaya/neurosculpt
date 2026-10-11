"""Batches of patient IDs that have a given image at both visits (base of the other data loaders)."""
import os
import random

import numpy as np

from brainpred import paths, splits
from brainpred.utils import io


class DataLoaderIds:
    """Iterate over batches of patient IDs.

    Args:
        image_sufix: image name pattern formatted with ``(id, hunt - 3)``, e.g. ``paths.INAME_PREPROCESSED``.
            Only patients having this image at HUNT3 and HUNT4 are kept.
        batch_size: number of IDs per batch.
        shuffle: shuffle the IDs at creation and after every full pass.
        nb_patients: keep only this many patients (``<= 0``: all).
        exclude_ids: IDs removed before selecting ``nb_patients``.
        start_from: index of the first selected patient.
        patient_ids: candidate IDs (default: ``splits.get_matching_ids()``).
    """

    def __init__(self, image_sufix, batch_size, shuffle=False, nb_patients=-1, exclude_ids=None, start_from=0,
                 patient_ids=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_sufix = image_sufix
        self.nb_patients = nb_patients
        self.exclude_ids = exclude_ids
        self.start_from = start_from
        self.iter_ids = None
        self.__load_patient_ids(patient_ids)

    def __load_patient_ids(self, candidate_ids):
        self.patient_ids = self.__get_patients_ids_with_requested_image(candidate_ids)

        if self.exclude_ids is not None:
            self.patient_ids = [x for x in self.patient_ids if x not in self.exclude_ids]

        if (self.start_from < 0) or (self.start_from >= len(self.patient_ids)):
            self.start_from = 0
        if ((self.nb_patients + self.start_from) > len(self.patient_ids)) or (self.nb_patients <= 0):
            self.nb_patients = len(self.patient_ids) - self.start_from

        self.patient_ids = self.patient_ids[self.start_from:self.start_from + self.nb_patients]

        if self.shuffle:
            self.shuffle_ids()

    def __len__(self):
        return int(np.ceil(self.nb_patients / self.batch_size))

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
            self.iter_ids = ret
            return ret

        if index < 0:
            index = self.__len__() + index
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.patient_ids))
        self.iter_ids = self.patient_ids[start:end]
        return self.iter_ids

    def __iter__(self):
        for i in range(0, self.nb_patients, self.batch_size):
            self.iter_ids = self.patient_ids[i:min(i + self.batch_size, len(self.patient_ids))]
            yield self.iter_ids
        if self.shuffle:
            self.shuffle_ids()
        self.iter_ids = None

    def shuffle_ids(self):
        random.shuffle(self.patient_ids)

    def get_custom_image_path_name(self, hunt, patient_id, custom_img_sufix):
        return os.path.join(paths.get_patient_folder(hunt, patient_id), custom_img_sufix.format(patient_id, hunt - 3))

    def get_image_path_name(self, hunt, patient_id):
        return self.get_custom_image_path_name(hunt, patient_id, self.image_sufix)

    def get_image_bmask_path_name(self, hunt, patient_id):
        return paths.get_brain_mask_path_name(hunt, patient_id, paths.REGISTRATION_TO_MNI)

    def __get_patients_ids_with_requested_image(self, candidate_ids):
        if candidate_ids is None:
            candidate_ids = splits.get_matching_ids()
        return [patient_id for patient_id in candidate_ids
                if io.verify_file(self.get_image_path_name(3, patient_id))
                and io.verify_file(self.get_image_path_name(4, patient_id))]
