# Mapping from the original research code

This repository was reorganised from the original thesis repository (`Master_Thesis-Brain_Prediction`).
This page maps old files to new ones and lists every behavioural change made during the cleanup.

## General changes

* `sys.path.append(...)` hacks and `load_module_from_path` were replaced by imports from the `brainpred` package.
* Hardcoded paths (`/home/...`, `/media/...`, `C:/...`, `/cluster/...`) and the `SERVER` flag were replaced by
  `configs/paths.yaml` (`brainpred.config`).
* The train/validation split (`get_matching_ids()[:621]`, `["08202"] + get_matching_ids()[621:]`) and
  `EXCLUDED_PATIENTS` moved to `configs/splits.yaml` (`brainpred.splits`).
* The 12 per-approach `constants.py` files were replaced by `brainpred.results.ResultsLayout`. Method folder names
  stay the same (`MGAN_paper_16`, `dfield_similar_patches`, ...), so old result folders can be copied to
  `<results_dir>/` unchanged.
* Pipelines run by commenting and uncommenting blocks became `--steps` options of the scripts.
* Spanish comments were translated and dead commented-out code was removed.

## Old → new

| Original | New |
|---|---|
| `project_images_loader.py` | `paths.py` (naming), `splits.py` (subjects), `config.py` (locations) |
| `utils/nifti_functions.py`, filesystem part of `utils/functions.py` | `utils/io.py` |
| `utils/util.py` | `utils/intensity.py`, `utils/metrics.py`, plotting in `utils/display.py` |
| `utils/functions.py` (only the ~25 helpers actually used), `utils/visualization.py` | `utils/display.py`, `utils/arrays.py` |
| `utils/util_freesurfer_segmentation.py`, `patches_controller.py`, `util_brain_metrics.py`, `util_deep_learning.py`, `util_machine_learning.py` | `utils/segmentation.py`, `patches.py`, `brain_metrics.py`, `deep_learning.py`, `ml.py` |
| `utils/dfield_applier.py` + `util_dfield_metrics.py` | `utils/dfield.py` |
| `preprocessing/prep_*.py`, `preprocessing.py` | `preprocessing/*.py`, `scripts/preprocess.py` |
| `preprocessing/registration_parameters_elastix/` (+ 2 duplicate copies) | `configs/elastix/` |
| `postprocessing/post_*.py`, `approaches/*/postprocess_images.py` (×8) | `postprocessing/steps.py`, `postprocessing/pipeline.py`, `scripts/postprocess.py` |
| `test_code/dfield_approach_elastix/register_elastix.py` | `registration/elastix.py`, `scripts/register_longitudinal.py` |
| `test_code/creator_2D_images/` | `preprocessing/slices_2d.py`, `scripts/make_2d_slices.py` |
| `statistics/patients/<kind>/<base>/features_{computer,loader}.py` | `features/<kind>/<base>.py`, `scripts/compute_features.py` |
| `approaches/dfield_*/*_dfields_creator.py` | `methods/dfield/<method>.py` (`similar_tissues_v2` is an option of `similar_tissues.py`) |
| `approaches/dfield_*/register_val_images.py` (×6) | `methods/dfield/apply.py` |
| `approaches/dfield_similar_images_registered/similar_images_registrations/` | `methods/dfield/similar_images_registered.py` |
| `approaches/{MGAN,MPGAN}_paper_16*/` + `test_code/{MGAN,MPGAN}/*_paper_16{,_seg_loss}/` | `methods/gan/` (`mgan/`, `mpgan/`, `losses.py`, `training.py`, `inference.py`, `variants.py`), `scripts/{train,predict}_gan.py` |
| `layers/Wavelet3D_16.py` | `methods/gan/layers/wavelet3d.py` (`filter_dtype` argument) |
| `data_loaders/data_loader_{patches,ids,complete}.py` | `data/{patches,ids,complete}.py` |
| `approaches/StarGAN_V2_sagital/` + `test_code/stargan-v2-master/brain_2D_mask/main.py` + `data_loaders/data_loaders_2D/` | `methods/stargan_v2/`, `data/stargan_2d.py`, `scripts/{train,predict}_stargan.py` |
| `approaches/diffusion_model/` | `methods/diffusion/` (2D→3D reconstruction only) |
| `approaches/*/compute_statistics(_raw).py` | `evaluation/statistics.py`, `scripts/evaluate.py` |
| `approaches/*/visualize_results.py`, `approaches/visualize_results{,_poster}.py` | `evaluation/figures.py`, `scripts/make_figures.py` |
| `statistics/**/*.ipynb` | `notebooks/analysis/` (outputs stripped, imports updated) |
| `test_code/MGAN/MGAN_paper_16/slurm/launch.slurm` | `slurm/train_gan.slurm` |
| `create_tensorflow_environment.txt` | `pyproject.toml`, README |

### Not carried over

* **Exploratory code** in `test_code/`: earlier GAN variants (`MGAN_paper`, `*_pix2pix*`, `MPGAN_paper*`,
  `*_with_mask`), the Reg-GAN, VoxelMorph, Genesis and MONAI experiments, `gan_example/`, `fastsurfer_test/`, the
  `approach_*` prototypes of the deformation-field methods, and the two other StarGAN variants.
* **One-off scripts**: `__copy_*.py`, `__change_images_name.py`, `expr/checkpoints/remove_400.py`.
* **Unused modules**: `layers/Wavelet3D.py`, `Wavelet3D_2.py`, the unused data loaders, and about 100 unused helpers
  in `utils/functions.py`.
* **Data and outputs**: all `.csv` files (patient-level data), `res/*.zip`, `.ckpt` files, slurm logs, and notebook
  outputs.

## Behavioural changes (bug fixes)

Preprocessing and postprocessing:
* `post_skull_stripping`: undefined `_path_name_out` whenever an output path was given.
* `save_registration_to_hunt3_freesurfer` overwrote the HUNT4→MNI transform. This step is unused by default.
* 2D slices: `save_registration_to_MNI_elastix` never registered when `verify=False`.
* `register_elastix` transform step referenced names that did not exist. It now writes `SEG_96` and `SEG_36` with
  the names the methods read.

Deformation-field methods:
* Patch-similarity loader: `if exclude_ids is None` before iterating over `exclude_ids` is now `is not None`.
  Results are unchanged because the tables only contain training subjects.
* `similar_patches`: the `save_nifti` call of the predicted dfield was commented out and is re-enabled.
* `similar_tissues`: a debug `break` limited the creator and apply steps to the first subject.
* Inter-patient whole-brain features: hunt=4 was hardcoded and the output folder didn't match the loader. It is now
  `--hunt` (default 3).
* Reference exclusion is now target IDs + validation set + excluded patients. It used to be the target list only;
  this is identical when the targets are the validation set.

GANs and StarGAN:
* `predict_gan`: a stray `break` predicted only the first subject per checkpoint.
* MPGAN raw statistics: a `break` left every row after the first subject at 0.
* MPGAN `unet3d.py`: `RANDOM_SEED` was undefined (only on an unused path). It is now 36.
* StarGAN prediction: the custom-style strategy passed `id` instead of `id_sty`, and BPF-based style selection
  passed the exclusion list to the wrong argument.
* StarGAN `make_2d_slices.py`: defaults to sagittal slices (`--axis 0`), which StarGAN v2 uses. The original
  notebook's last call produced axial slices.

Figures:
* Several per-method viewers showed HUNT3 twice, with the second one labelled HUNT4.
* `plot_data_correlation_df` crashed because of a wrong argument.

### Kept on purpose (thesis numbers depend on them)

* The MGAN generator applies sigmoid twice on its lower-scale heads.
* The MPGAN (tfa) discriminator normalises on `axis=-1` with channels-first data.
* The mean dfield divides by `nb_images` even when subjects are excluded.
* Inter-patient absolute-difference features always use HUNT3 volumes.
* StarGAN v2 was trained on the first 620 subjects. The 3D methods use 621.
