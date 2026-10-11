
import numpy as np

def obtain_wm_gm_mask(seg):
    seg_wm_gm = np.zeros_like(seg)
    seg_wm_gm[(seg == 2) | (seg == 3)] = 1
    return seg_wm_gm


def compute_bpf(seg_csf_wm_gm, bmask):
    seg_wm_gm = obtain_wm_gm_mask(seg_csf_wm_gm)
    seg_wm_gm[bmask == 0] = 0 # seg_csf_wm_gm should be contained in the brain mask
    bpf = np.sum(seg_wm_gm)/np.sum(bmask)
    return bpf