"""Output layout shared by every prediction method.

Replaces the per-approach ``constants.py`` files of the original code. For a method
named ``m``, outputs are written to::

    <results_dir>/m/results/<id>/<id>_0_<TYPE>_<extra>.nii.gz                 raw predictions
    <results_dir>/m/results_postprocessed/<id>/<id>_0_<TYPE>_<extra>.nii.gz   after postprocessing
    <results_dir>/m/statistics[_postprocessed]/<name>.csv                     evaluation tables
    <results_dir>/m/dfields/<id>/...                                          (deformation-field methods)

``<extra>`` encodes the method's hyper-parameters (e.g. checkpoint, number of
reference images), joined with ``_``.
"""
from brainpred.config import get_config

INAME_RESULT_IMAGE = "{0}_0_{1}_{2}.nii.gz"


class ResultsLayout:
    def __init__(self, method_name, dfield_name_pattern=None):
        """
        Args:
            method_name: name of the method's output folder.
            dfield_name_pattern: format string for predicted deformation fields
                (deformation-field methods only), receiving ``id`` and ``*extra_info``.
        """
        self.method_name = method_name
        self.dfield_name_pattern = dfield_name_pattern

    def get_images_path(self):
        return f"{get_config().results_dir}/{self.method_name}"

    def get_dfield_path_name(self, id, *extra_info):
        return f"{self.get_images_path()}/dfields/{id}/{self.dfield_name_pattern.format(id, *extra_info)}"

    def get_result_patient_folder(self, id, postprocessed=False):
        folder = "results_postprocessed" if postprocessed else "results"
        return f"{self.get_images_path()}/{folder}/{id}"

    def get_result_image_path_name(self, id, type_img, extra_info, postprocessed=False):
        """Possible type_img: T1, BMASK, BMASK_NCRB, DFIELD, SEG_96, SEG_36, SEG_3 and their _POST variants."""
        extra_info_str = "_".join(str(info) for info in extra_info)
        return f"{self.get_result_patient_folder(id, postprocessed)}/{INAME_RESULT_IMAGE.format(id, type_img, extra_info_str)}"

    def get_statistics_path_name(self, statistics_name, postprocessed=False):
        folder = "statistics_postprocessed" if postprocessed else "statistics"
        return f"{self.get_images_path()}/{folder}/{statistics_name}.csv"
