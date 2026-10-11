"""Shared helpers for the patient feature tables: folders, segmentation loading and table loaders."""
import numpy as np
import pandas as pd

from brainpred import paths
from brainpred.utils import io
from brainpred.utils import segmentation as ufs

WHOLE_BRAIN_TABLE_NAME = "whole_brains_stats"


def get_features_folder(*parts):
    """``<work_dir>/features/<parts>``: where every feature table is written and read."""
    return paths.get_work_folder("features", *parts)


def save_table(df, path_name, index=False):
    io.create_folders(path_name.rsplit("/", 1)[0])
    df.to_csv(path_name, index=index)


# ---- segmentations
def load_segmentation(id, hunt, nb_tissues, remove_cerebellum=False):
    """Load the ``nb_tissues`` segmentation; optionally remove the cerebellum using the 36-tissue one."""
    seg, _ = io.load_nifti(paths.get_segmentation_path_name(hunt, id, nb_tissues))
    if remove_cerebellum:
        seg_36, _ = io.load_nifti(paths.get_segmentation_path_name(hunt, id, 36))
        return ufs.remove_cerebellum(seg, seg_36, deep_copy=False)
    return seg


def merge_segmentation_96(seg_96, nb_tissues):
    """Reduce a 96-tissue segmentation to 3 tissues (without cerebellum) or 36 tissues."""
    if nb_tissues == 3:
        return ufs.remove_cerebellum(ufs.merge_seg96_to_seg3(seg_96), seg_96, deep_copy=False)
    if nb_tissues == 36:
        return ufs.merge_seg96_to_seg36(seg_96)
    return seg_96


def load_segmentation_from_96(id, hunt, nb_tissues):
    seg, _ = io.load_nifti(paths.get_segmentation_path_name(hunt, id, nb_tissues=96))
    return merge_segmentation_96(seg, nb_tissues)


# ---- table loaders
class WholeBrainTableLoader:
    """One CSV with one row per patient (``patient_id`` + whole-brain metrics)."""

    def __init__(self, csv_path_name):
        self.path = csv_path_name
        self.df = pd.read_csv(self.path)

    def get_df(self, columns=None, remove_columns=False, return_patient_id=True, id_patients=None, remove_patients=True):
        """Select ``columns`` (or all but them with ``remove_columns``); drop ``id_patients``
        (or keep only them with ``remove_patients=False``)."""
        if columns is None:
            columns = self.df.columns

        if remove_columns:
            _columns = [c for c in self.df.columns if c not in columns]
        else:
            _columns = columns

        if return_patient_id:
            _columns.insert(0, self.df.columns[0])

        _df = self.df[_columns]
        if id_patients is not None:
            id_patients = [int(p) for p in id_patients]
            indices = self.df[self.df['patient_id'].isin(id_patients)].index
            if remove_patients:
                _df = _df.drop(indices)
            else:
                _df = _df.loc[indices]
        return _df


class TissueTableLoader:
    """One CSV per patient (rows = tissue ids, columns = metrics), flattened to
    one row per patient with columns ``<tissue_id>_<metric>``."""

    def __init__(self, folder, ids):
        self.path = folder
        self.ids = ids
        self.columns = self._prepare_new_columns()
        self.df = self._read_statistics()

    def _load_csv(self, id):
        return pd.read_csv(f'{self.path}/{id}.csv')

    def _prepare_new_columns(self):
        _df = self._load_csv(self.ids[0])
        _columns = _df.columns.tolist()[1:]
        _tissue_ids = _df['tissue_id'].tolist()

        new_columns = ['patient_id']
        for _tissue_id in _tissue_ids:
            for _column in _columns:
                new_columns.append(f'{_tissue_id}_{_column}')
        return new_columns

    def _read_statistics(self):
        _data = np.zeros((len(self.ids), len(self.columns)), np.float32)
        for i, id in enumerate(self.ids):
            _patient_data = self._load_csv(id).values[:, 1:]
            _data[i, 0] = int(id)
            _data[i, 1:] = _patient_data.flatten()
        return pd.DataFrame(_data, columns=self.columns)

    def get_df(self, tissue_ids=None, metrics_name=None, remove_tissues=False, remove_metrics=False, return_patient_id=True, id_patients=None, remove_patients=True):
        """Select tissues and metrics (or all but them with ``remove_*``); drop ``id_patients``
        (or keep only them with ``remove_patients=False``)."""
        if tissue_ids is not None:
            _columns_tissue_ids = [c for c in self.columns[1:] if int(c.split('_')[0]) in tissue_ids]
        else:
            _columns_tissue_ids = self.columns[1:]

        if metrics_name is not None:
            _columns_metrics = [c for c in self.columns[1:] if '_'.join(c.split('_')[1:]) in metrics_name]
        else:
            _columns_metrics = self.columns[1:]

        if remove_tissues:
            _columns = [c for c in self.columns if c not in _columns_tissue_ids]
        else:
            _columns = _columns_tissue_ids
        if remove_metrics:
            _columns = [c for c in _columns if c not in _columns_metrics]
        else:
            _columns = [c for c in _columns if c in _columns_metrics]

        if return_patient_id:
            _columns.insert(0, self.columns[0])

        _df = self.df[_columns]
        if id_patients is not None:
            id_patients = [int(p) for p in id_patients]
            indices = self.df[self.df['patient_id'].isin(id_patients)].index
            if remove_patients:
                _df = _df.drop(indices)
            else:
                _df = _df.loc[indices]
        return _df


class PairwiseSimilarityLoader:
    """Symmetric inter-patient similarity tables.

    ``<folder>/<id>.csv`` holds the similarity of ``id`` against every patient after it in
    ``ids`` (upper triangle); the full ``(n_ids, n_ids, n_metrics)`` array is rebuilt here.
    """

    def __init__(self, folder, ids, allow_missing=False):
        self.path = folder
        self.ids = ids
        self.allow_missing = allow_missing
        self.similarity_keys = self._load_similarity_keys()
        self.similarity_np = self._create_similarity_np()

    def _get_similarity_csv_path_name(self, id):
        return f"{self.path}/{id}.csv"

    def _load_similarity_keys(self):
        # similarity keys without the id
        df = pd.read_csv(self._get_similarity_csv_path_name(self.ids[0]))
        return list(df.columns)[1:]

    def _create_similarity_np(self):
        _similarity_np = np.zeros((len(self.ids), len(self.ids), len(self.similarity_keys)))
        for index, id in enumerate(self.ids[:-1]):
            path_name = self._get_similarity_csv_path_name(id)
            if self.allow_missing and not io.verify_file(path_name):
                continue
            df_csv = pd.read_csv(path_name)
            _similarity_np[index, index+1:, :] = df_csv.values[:, 1:]

        # mirror the upper triangle
        _similarity_np += np.triu(_similarity_np.transpose(2, 0, 1), k=1).transpose(2, 1, 0)
        return _similarity_np

    def get_similarity_metric(self, id1, id2, metric):
        index1 = self.ids.index(id1)
        index2 = self.ids.index(id2)
        index_metric = self.similarity_keys.index(metric)
        return self.similarity_np[index1, index2, index_metric]

    def get_patient_similarity_table(self, id, exclude_ids=None):
        """Similarity of ``id`` against every other patient (without itself and ``exclude_ids``)."""
        index = self.ids.index(id)
        df = pd.DataFrame(self.similarity_np[index], columns=self.similarity_keys)
        df.insert(0, 'id', self.ids)
        df = df[df['id'] != id]
        if exclude_ids is not None:
            df = df[~df['id'].isin(exclude_ids)]
        return df

    def get_sorted_similarity_table(self, id, sim_metric, nb_images=-1, exclude_ids=None):
        """The ``nb_images`` most similar patients by ``sim_metric`` with an L1-normalised ``weight`` column."""
        patient_sim_table = self.get_patient_similarity_table(id, exclude_ids=exclude_ids)
        df_metrics = patient_sim_table.sort_values(sim_metric, ascending=False)
        if nb_images > 0:
            df_metrics = df_metrics.head(nb_images)
        df_metrics['weight'] = df_metrics[sim_metric] / df_metrics[sim_metric].sum()
        return df_metrics

    def get_metric_similarity_table(self, sim_metric):
        return pd.DataFrame(self.similarity_np[:, :, self.similarity_keys.index(sim_metric)], columns=self.ids)

    # name used by the whole-brain loader in the original code
    get_all_images_similarity_table = get_metric_similarity_table
