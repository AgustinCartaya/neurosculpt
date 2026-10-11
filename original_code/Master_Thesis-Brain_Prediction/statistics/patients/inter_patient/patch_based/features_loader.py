
import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../../'))

import numpy as np
import pandas as pd

class FeaturesLoader:
    def __init__(self, nb_tissues):
        self.path = f'{current_path}/features/tissues_{nb_tissues}'
        
    def get_patient_table(self, id, exclude_ids=None):
        df = self.__load_csv(id)
        if exclude_ids is None:
            exclude_ids = [int(id) for id in exclude_ids]
            df = df[~df['id'].isin(exclude_ids)]
        return df

    def __load_csv(self, id):
        return pd.read_csv(f'{self.path}/{id}.csv')

   