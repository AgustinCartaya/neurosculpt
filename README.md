# brainpred — Predicting future brain MRI from longitudinal scans

Code of the master thesis *"<THESIS TITLE>"* (Agustín Cartaya, 2024).

Given a subject's T1-weighted brain MRI at a first visit (HUNT3), the goal is to predict
what their brain will look like ~11 years later (HUNT4). Two families of methods are compared:

| Family | Methods | Idea |
|---|---|---|
| **Deformation-field** | `mean`, `similar_images`, `similar_images_registered`, `similar_tissues`, `similar_patches` | Build a HUNT3→HUNT4 deformation field for the new subject from the observed longitudinal deformations of the most similar reference subjects (globally, per tissue or per patch), then warp the subject's scan. |
| **Generative** | MGAN, MPGAN (3D, TensorFlow), StarGAN v2 (2D sagittal, PyTorch), latent diffusion (external) | Learn the HUNT3→HUNT4 mapping directly. |

Predictions are evaluated against the real HUNT4 scan with SSIM, NCC, PSNR, brain parenchymal fraction (BPF)
and Dice of FastSurfer tissue segmentations.

> **Data:** this repository contains no imaging or patient data. The HUNT data can only be obtained from the
> HUNT Research Centre, see [data/README.md](data/README.md). The code can be run on any longitudinal T1 dataset
> organised as described in [docs/data.md](docs/data.md).

## Repository layout

```
configs/            paths.example.yaml, splits.example.yaml, elastix/ (registration parameter files)
src/brainpred/      the Python package
  config.py         loads configs/paths.yaml (all machine-specific paths)
  paths.py          file naming conventions of the dataset
  splits.py         subject lists and train / validation split
  results.py        output layout shared by all methods
  preprocessing/    SynthSR, MNI registration, skull stripping, normalization, FastSurfer, 2D slices
  registration/     longitudinal HUNT3 -> HUNT4 elastix registration
  features/         per-subject and inter-subject features (similarity tables, BPF, Dice, ...)
  methods/
    dfield/         deformation-field-based prediction methods
    gan/            MGAN / MPGAN (layers, models, losses, training, inference)
    stargan_v2/     2D StarGAN v2 (CC BY-NC 4.0, derived from clovaai/stargan-v2)
    diffusion/      helpers for externally produced diffusion-model predictions
  data/             data loaders for training
  postprocessing/   postprocessing of predicted images (same pipeline as preprocessing)
  evaluation/       statistics and figures
  utils/            I/O, intensity, metrics, segmentation labels, patches, plotting
scripts/            command-line entry points (one per pipeline stage, all support --help)
notebooks/analysis/ statistical analysis notebooks (outputs stripped)
slurm/              SLURM template for GAN training
docs/               pipeline, data layout, mapping from the original code
```

## Installation

Python 3.10–3.11. The base package covers preprocessing, deformation-field methods and evaluation;
the GANs need extra dependencies.

```bash
conda create -n brainpred python=3.10 -y && conda activate brainpred
pip install -e .            # base
pip install -e ".[tf]"      # + MGAN / MPGAN (TensorFlow 2.15, tensorflow-addons)
pip install -e ".[torch]"   # + StarGAN v2
```

External tools used by preprocessing and postprocessing (must be on `PATH`):
[FreeSurfer 7.x](https://surfer.nmr.mgh.harvard.edu/) with SynthSR/SynthStrip (`mri_synthsr`, `mri_synthstrip`, `mri_robust_register`,
`mri_label2vol`) and [FastSurfer](https://github.com/Deep-MI/FastSurfer) through Docker (`deepmi/fastsurfer`).

## Configuration

```bash
cp configs/paths.example.yaml configs/paths.yaml     # where data, results, checkpoints... live
cp configs/splits.example.yaml configs/splits.yaml   # train / validation split
```

Both files are git-ignored. `BRAINPRED_CONFIG` / `BRAINPRED_SPLITS` can point to other files.

## Running the pipeline

Every script accepts `--help`, `--ids ...` or `--split train|validation|all`. See [docs/pipeline.md](docs/pipeline.md)
for the full description and the exact commands that reproduce each thesis result.

```bash
# 1. Preprocessing (FreeSurfer + FastSurfer)
python scripts/preprocess.py --split all --steps synthsr,register_mni,skull_strip,normalize,segment,seg_to_patient,copy_seg,merge_tissues,mask_ncrb

# 2. Longitudinal registration HUNT3 -> HUNT4 (deformation fields of the reference subjects)
python scripts/register_longitudinal.py --split all

# 3. Features / inter-subject similarity
python scripts/compute_features.py cross_sectional_tissue --help   # one sub-command per feature kind

# 4a. Deformation-field methods
python scripts/predict_dfield.py --method similar_patches

# 4b. Generative methods
python scripts/train_gan.py --model mgan
python scripts/predict_gan.py --model mgan --checkpoints 46
python scripts/make_2d_slices.py --split all --steps register,transform,normalize,inverse,slices
python scripts/train_stargan.py
python scripts/predict_stargan.py --step 65200 --strategy best

# 5. Postprocessing of the predictions, evaluation and figures
python scripts/postprocess.py --method MGAN_paper_16 --extra 46
python scripts/evaluate.py --method MGAN_paper_16 --extra 46 --postprocessed --type-suffix _POST --metrics ssim,ncc,psnr,bpf,dice
python scripts/make_figures.py --help
```

## Status

This repository is a cleaned-up version of the original research code: files were reorganised into a package,
hardcoded paths were replaced by configuration, duplicated code was merged and run-by-commenting blocks became
command-line options. Algorithms and hyper-parameters were kept unchanged. The few bugs fixed during the cleanup are
listed in [docs/migration.md](docs/migration.md). The cleaned code has not been re-run end-to-end on the HUNT data.

## License

MIT, see [LICENSE](LICENSE), except `src/brainpred/methods/stargan_v2/` and `src/brainpred/data/stargan_2d.py`,
which derive from [StarGAN v2](https://github.com/clovaai/stargan-v2) and are licensed under CC BY-NC 4.0.

## Citation

See [CITATION.cff](CITATION.cff).
