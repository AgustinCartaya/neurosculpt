"""Step 6: merge the 36-label FastSurfer segmentation into 3 tissues (CSF, GM, WM) -> ``SEG_3``."""
from brainpred import paths
from brainpred.utils import io
from brainpred.utils import segmentation as ufs


def save_3_tissues_segmentation(hunt, id, verify=True):
    seg_3_path_name = paths.get_segmentation_path_name(hunt, id, nb_tissues=3)
    if verify and io.verify_file(seg_3_path_name):
        print(f"3 tissues already done for HUNT{hunt} {id}")
        return True

    seg, aff = io.load_nifti(paths.get_segmentation_path_name(hunt, id, nb_tissues=36))
    io.save_nifti(ufs.merge_seg36_to_seg3(seg), aff, seg_3_path_name)
