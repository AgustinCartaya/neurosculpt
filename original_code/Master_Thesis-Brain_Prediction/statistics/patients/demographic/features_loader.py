

import sys
import os
current_path = os.path.dirname(__file__)

import pandas as pd


class FeaturesLoader:
    def __init__(self):
        self.path = f'{current_path}/features/demographic'
        self.df = self.__load_csv()
          
    def get_df(self, columns=None, remove_columns=False, return_patient_id=True, remove_patients=None):
        # _columns = self.columns
        if columns is None:
            columns = self.df.columns
            
        if remove_columns:
            _columns = self.df.columns
            _columns = [c for c in _columns if c not in columns]
        else:
            _columns = columns

        if return_patient_id:
            _columns.insert(0, self.df.columns[0])
       
        _df = self.df[_columns]
        if remove_patients is not None:
            remove_patients = [int(p) for p in remove_patients]
            indices = self.df[self.df['patient_id'].isin(remove_patients)].index
            _df = _df.drop(indices)

        return _df

    def __load_csv(self):
        return pd.read_csv(f'{self.path}.csv')