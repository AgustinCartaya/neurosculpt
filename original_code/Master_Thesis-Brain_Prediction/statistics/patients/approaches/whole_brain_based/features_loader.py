
import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../../'))

sys.path.append(base_path)

import pandas as pd


class FeaturesLoader:
    def __init__(self, approach):
        self.path = f'{current_path}/features/{approach}/whole_brains_stats'
        self.df = self.__load_csv()
          
    def get_df(self, columns=None, remove_columns=False, return_patient_id=True, id_patients=None, remove_patients=True):
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
        if id_patients is not None:
            id_patients = [int(p) for p in id_patients]
            indices = self.df[self.df['patient_id'].isin(id_patients)].index
            if remove_patients:
                _df = _df.drop(indices)
            else:
                _df = _df.loc[indices]

        return _df

    def __load_csv(self):
        return pd.read_csv(f'{self.path}.csv')