import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../../'))

sys.path.append(base_path)

import numpy as np
import pandas as pd
from tqdm import tqdm
import nibabel as nib

import project_images_loader as pil
import utils.nifti_functions as nfc
import utils.functions as fc



class FeaturesLoader():
    def __init__(self, ids, hunt=3):
        self.path = f'{current_path}/features_hunt{hunt}'
        self.ids = ids

        self.similarity_keys = self._load_similarity_keys()
        self.__create_similarity_np()
        # self.__save_similarity_np_as_nifti()

    def _read_available_ids(self):
        possible_files = os.listdir(self.path)
        patient_folders = [f[:-4] for f in possible_files if f.endswith(".csv")]
        return patient_folders

    def _get_similarity_csv_path_name(self, id):
        return f"{self.path}/{id}.csv"
    
    def _load_similarity_keys(self):
        df = pd.read_csv(self._get_similarity_csv_path_name(self.ids[0]))
        return list(df.columns)[1:]

    def __create_similarity_np(self):
        # create similarity 3D table 
        self.similarity_np = np.zeros((len(self.ids), len(self.ids), len(self.similarity_keys)))
        # print(len(self.ids))

        for index, id in enumerate(self.ids[:-1]):
            if fc.verify_file(self._get_similarity_csv_path_name(id)):
                df_csv = pd.read_csv(self._get_similarity_csv_path_name(id))
                self.similarity_np[index, index+1:, :] = df_csv.values[:, 1:]

        self.similarity_np += np.triu(self.similarity_np.transpose(2, 0, 1), k=1).transpose(2, 1, 0)

    def get_similarity_metric(self, id1, id2, metric):
        index1 = self.ids.index(id1)
        index2 = self.ids.index(id2)
        index_metric = self.similarity_keys.index(metric)
        return self.similarity_np[index1, index2, index_metric]

    def get_patient_similarity_table(self, id, exclude_ids=None):
        index = self.ids.index(id)
        # remove curretn element
        sim_elements = self.similarity_np[index]

        df = pd.DataFrame(sim_elements, columns=self.similarity_keys)
        df.insert(0, 'id', self.ids)

        df = df[df['id'] != id]
        if exclude_ids is not None:
            # exclude_ids_int = [int(excluded_id) for excluded_id in exclude_ids]
            df = df[~df['id'].isin(exclude_ids)]
        return df

    def get_sorted_similarity_table(self, id, sim_metric, nb_images=-1, exclude_ids=None):
        patient_sim_table = self.get_patient_similarity_table(id, exclude_ids=exclude_ids)
        df_metrics = patient_sim_table.sort_values(sim_metric, ascending=False)
        if nb_images > 0:
            df_metrics = df_metrics.head(nb_images)
        df_metrics['weight'] = df_metrics[sim_metric] / df_metrics[sim_metric].sum()
        return df_metrics
    
    def get_all_images_similarity_table(self, sim_metric):
        df = pd.DataFrame(self.similarity_np[:, :, self.similarity_keys.index(sim_metric)], columns=self.ids)
        return df
    
# np.set_printoptions(precision=3)

# matchin_ids = pil.get_matching_ids()
# ips = SimilarityTablesReader(matchin_ids)
# # # print(ips.get_similarity(ips.ids[0], ips.ids[1], 'rms'))
# _df = ips.get_patient_similarity_table(matchin_ids[-1])
# print(_df.head(10))    

