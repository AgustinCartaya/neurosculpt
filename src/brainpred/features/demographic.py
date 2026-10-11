"""Loader for the demographic table (one row per patient: ``patient_id``, gender, ages at HUNT3 / HUNT4, ...)."""
from brainpred.features.common import WholeBrainTableLoader, get_features_folder


def get_table_path_name():
    return f'{get_features_folder("demographic")}/demographic.csv'


class FeaturesLoader(WholeBrainTableLoader):
    def __init__(self):
        super().__init__(get_table_path_name())

    def get_df(self, columns=None, remove_columns=False, return_patient_id=True, remove_patients=None):
        """Same as :meth:`WholeBrainTableLoader.get_df` but ``remove_patients`` is the list of ids to drop."""
        return super().get_df(columns=columns, remove_columns=remove_columns, return_patient_id=return_patient_id,
                              id_patients=remove_patients, remove_patients=True)
