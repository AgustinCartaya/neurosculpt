
import os

current_path = os.path.dirname(__file__)

import numpy as np
import pandas as pd

class FeaturesLoader:
    def __init__(self, approach, nb_tissues, ids):
        self.path = f'{current_path}/features/{approach}/tissues_{nb_tissues}'
        self.ids = ids

        self.columns = self.__prepare_new_columns()
        self.df = self.__read_statistics()
         
    def __prepare_new_columns(self):
        _df = self.__load_csv(self.ids[0])
        _columns = _df.columns.tolist()[1:]
        _tissue_ids = _df['tissue_id'].tolist()

        new_columns = ['patient_id']
        for _tissue_id in _tissue_ids:
            for _column in _columns:
                new_columns.append(f'{_tissue_id}_{_column}') 
        return new_columns
        

    def __read_statistics(self):
        _data = np.zeros((len(self.ids), len(self.columns)), np.float32)
        for i, id in enumerate(self.ids):
            _patient_df = self.__load_csv(id)
            _patient_data = _patient_df.values[:, 1:]

            _data[i, 0] = int(id)
            _data[i, 1:] = _patient_data.flatten()
        
        return pd.DataFrame(_data, columns=self.columns)

        
    def get_df(self, tissue_ids=None, metrics_name=None, remove_tissues=False, remove_metrics=False, return_patient_id=True, id_patients=None, remove_patients=True):
        # remove tissues
        if tissue_ids is not None:
            _columns_tissue_ids = [c for c in self.columns[1:] if int(c.split('_')[0]) in tissue_ids]
        else:
            _columns_tissue_ids = self.columns[1:]

        # remove columns
        if metrics_name is not None:
            _columns_metrics = [c for c in self.columns[1:] if '_'.join(c.split('_')[1:]) in metrics_name]
        else:
            _columns_metrics = self.columns[1:]

        # _columns = self.columns
        if remove_tissues:
            _columns = self.columns
            _columns = [c for c in _columns if c not in _columns_tissue_ids]
            if remove_metrics:
                _columns = [c for c in _columns if c not in _columns_metrics]
            else:
                _columns = [c for c in _columns if c in _columns_metrics]
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

    def __load_csv(self, id):
        return pd.read_csv(f'{self.path}/{id}.csv')

   