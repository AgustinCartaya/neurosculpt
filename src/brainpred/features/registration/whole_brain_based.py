"""Registration whole-brain features: BPF and similarity between the HUNT3 image registered to HUNT4 and the real HUNT4."""
from brainpred import paths
from brainpred.features.common import WHOLE_BRAIN_TABLE_NAME, WholeBrainTableLoader, get_features_folder
from brainpred.features.longitudinal.whole_brain_based import compute_table
from brainpred.utils import intensity, io
from brainpred.utils import segmentation as ufs


def get_table_path_name():
    return f'{get_features_folder("registration", "whole_brain_based")}/{WHOLE_BRAIN_TABLE_NAME}.csv'


def load_img(id, hunt):
    """HUNT3: longitudinal elastix registration to HUNT4 (segmentation merged to 3 tissues); HUNT4: MNI image."""
    if hunt == 3:
        img, _ = io.load_nifti(paths.get_longitudinal_registration_path_name(id, "T1"))
        seg, _ = io.load_nifti(paths.get_longitudinal_registration_path_name(id, "SEG_96"))
        seg = ufs.merge_seg96_to_seg3(seg)
    else:
        img, _ = io.load_nifti(paths.get_registered_to_template_image_path_name(hunt, id, registration_to=paths.REGISTRATION_TO_MNI))
        seg, _ = io.load_nifti(paths.get_segmentation_path_name(hunt, id, 3))
    bmask_ncereb, _ = io.load_nifti(paths.get_brain_mask_path_name(hunt, id, paths.REGISTRATION_TO_MNI, no_cerebellum=True))
    img = intensity.skull_stripping_and_normalize(img, seg)
    return img, seg, bmask_ncereb


def compute_patient_metrics_for_all():
    compute_table(load_img, get_table_path_name())


class FeaturesLoader(WholeBrainTableLoader):
    def __init__(self):
        super().__init__(get_table_path_name())
