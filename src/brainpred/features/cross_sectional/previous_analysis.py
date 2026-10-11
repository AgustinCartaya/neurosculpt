"""Loader for the CSV tables of the previous HUNT MRI analysis (patients identified as ``94100000<id>``)."""
import pandas as pd


class FeaturesLoader:
    def __init__(self, path, name):
        self.path = path
        self.name = name

    def get_df(self, id_column="", ids=None):
        _df = self._load_csv()
        if ids is not None:
            str_ids = [int("94100000{0}".format(id)) for id in ids]
            _df = _df[_df[id_column].isin(str_ids)]
        return _df

    def _load_csv(self):
        return pd.read_csv(f'{self.path}/{self.name}.csv')
