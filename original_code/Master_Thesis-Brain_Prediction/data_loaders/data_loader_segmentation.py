import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../'))
sys.path.append(base_path)

import numpy as np
import utils.nifti_functions as nfc
import random
import utils.functions as fc
# from data_loaders.data_loader_ids import DataLoaderIds

class DataLoaderComplete:

    def __init__(self, image_sufix, batch_size, image_size, channels_first=False, shuffle=False, nb_patients=-1, shuffle_before_selecting=False, server=True, data_type=np.float32, exclude_ids=None, start_from=0):
        self.batch_size = batch_size
        self.image_size = image_size
        self.channels_first = channels_first
        self.data_type = data_type
        
        self.ids_loader = DataLoaderIds(image_sufix=image_sufix, 
                                        batch_size=batch_size, 
                                        shuffle=shuffle, 
                                        nb_patients=nb_patients, 
                                        shuffle_before_selecting=shuffle_before_selecting, 
                                        server=server,
                                        exclude_ids=exclude_ids,
                                        start_from=start_from)