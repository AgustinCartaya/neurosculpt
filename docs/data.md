# Data layout

## Input

`data_dir` (see `configs/paths.yaml`) contains one folder per visit and one folder per subject:

```
<data_dir>/
  HUNT3/<id>/<id>_0_T1_RAW.nii      first visit  (or <id>_0_T1_BFC.nii if bias-field corrected)
  HUNT4/<id>/<id>_1_T1_RAW.nii      second visit
```

Files are named `<id>_<visit>_<TYPE>.<ext>` with `visit = 0` for HUNT3 and `1` for HUNT4.
Subjects with both visits are used (`brainpred.splits.get_matching_ids`).

## Files written by preprocessing (`scripts/preprocess.py`)

| Type | Step | Description |
|---|---|---|
| `T1_SYNTHSR.nii` | synthsr | SynthSR-resampled T1 |
| `T1_MNI.nii`, `DFIELD_MNI.lta` | register_mni | rigid registration to MNI (HUNT4 is registered to the registered HUNT3) |
| `BMASK_MNI.nii` | skull_strip | SynthStrip brain mask |
| `T1_PREP.nii` | normalize | skull-stripped T1 scaled to [0, 1]; **the input of every method** |
| `SEG_96.nii.gz`, `SEG_36.nii.gz` | segment, seg_to_patient, copy_seg | FastSurfer parcellation (aparc+aseg, aseg) |
| `SEG_3.nii.gz` | merge_tissues | CSF / GM / WM |
| `BMASK_NCRB_MNI.nii.gz` | mask_ncrb | brain mask without cerebellum (used for BPF) |

## Intermediate artefacts (`work_dir`)

```
<work_dir>/
  longitudinal_registrations/   <id>_0_{T1,DFIELD,SEG_96,SEG_36,BMASK}_0038.nii.gz   (scripts/register_longitudinal.py)
  features/<kind>/...           feature and similarity tables (CSV)                    (scripts/compute_features.py)
  prep_mni_3D/, slices_2D_*/    affine-registered volumes and PNG slices for StarGAN    (scripts/make_2d_slices.py)
```

## Results (`results_dir`)

Every method writes to its own folder (`brainpred.results.ResultsLayout`):

```
<results_dir>/<method>/
  results/<id>/<id>_0_<TYPE>_<params>.nii.gz                raw prediction (T1, SEG_96, BMASK_NCRB, ...)
  results_postprocessed/<id>/<id>_0_<TYPE>_POST_<params>... after scripts/postprocess.py
  statistics[_postprocessed]/<metric>.csv                   scripts/evaluate.py
  dfields/<id>/...                                          predicted deformation fields (dfield methods)
<results_dir>/figures/                                      scripts/make_figures.py --save
```

`<params>` encodes the method's hyper-parameters, e.g. `96_10` (96 tissues, 10 reference subjects) or `46`
(GAN checkpoint).

## External files

* MNI ICBM152 2009c nonlinear symmetric template (`mni_template_dir`), needed for 2D slice creation.
* FreeSurfer `license.txt` (`freesurfer_license_dir`), needed by FastSurfer.
* Models Genesis weights `Genesis_Chest_CT.h5` in `<checkpoints_dir>/genesis/` (MPGAN perceptual loss),
  from https://github.com/MrGiovanni/ModelsGenesis.
