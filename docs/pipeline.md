# Pipeline and reproduction of the thesis results

All scripts are run from the repository root and accept `--help`. Subject selection is `--ids ...` or
`--split train|validation|all` (see `configs/splits.yaml`). Every stage reads and writes the files described in
[data.md](data.md), so stages can be run on different machines.

```
raw T1 ─► 1 preprocess ─► T1_PREP, SEG_*, BMASK_*
                │
                ├─► 2 register_longitudinal ─► HUNT3→HUNT4 deformation fields of the reference subjects
                ├─► 3 compute_features ──────► similarity tables (which reference subjects resemble the target)
                │
                ├─► 4a predict_dfield ─┐
                ├─► 4b train/predict_gan, train/predict_stargan, (external diffusion model)
                │                      │
                └──────────────────────┴─► 5 postprocess ─► 6 evaluate ─► 7 make_figures / notebooks
```

## 1. Preprocessing

```bash
python scripts/preprocess.py --split all \
  --steps synthsr,register_mni,skull_strip,normalize,segment,seg_to_patient,copy_seg,merge_tissues,mask_ncrb
```

`segment` runs FastSurfer in Docker (GPU). The other steps need FreeSurfer. When they run on different machines,
use `--seg-mgz-folder` / `--seg-nii-folder` and copy those folders between machines.

## 2. Longitudinal registration

```bash
python scripts/register_longitudinal.py --split all --steps register,transform
```

## 3. Features

```bash
python scripts/compute_features.py cross_sectional_tissue --nb-tissues 3 96
python scripts/compute_features.py inter_patient_tissue --nb-tissues 3 96
python scripts/compute_features.py inter_patient_patch --nb-tissues 96
python scripts/compute_features.py cross_sectional_whole_brain              # StarGAN reference selection
```

`compute_features.py <kind> --help` lists each kind's options. The remaining kinds (longitudinal, registration,
approaches) are only used by the analysis notebooks.

## 4a. Deformation-field methods

The defaults reproduce the configurations of the final comparison:

| Method | Command | Result |
|---|---|---|
| Mean dfield | `predict_dfield.py --method mean` | `dfield_mean/…_T1_100` |
| Similar images | `predict_dfield.py --method similar_images` | `…_T1_DICE_MEAN_100` |
| Similar images, registered | `predict_dfield.py --method similar_images_registered` | `…_T1_BSP_FAST_5` |
| Similar tissues | `predict_dfield.py --method similar_tissues` | `…_T1_96_10` |
| Similar patches | `predict_dfield.py --method similar_patches` | `…_T1_96_10` |

## 4b. Generative methods

| Model | Training | Prediction (final checkpoint) |
|---|---|---|
| MGAN | `train_gan.py --model mgan` | `predict_gan.py --model mgan --checkpoints 46` |
| MPGAN | `train_gan.py --model mpgan` | `predict_gan.py --model mpgan --checkpoints 28` |
| MPGAN + seg. loss | `train_gan.py --model mpgan_seg` | `predict_gan.py --model mpgan_seg --checkpoints 52` |
| MGAN + seg. loss | `train_gan.py --model mgan_seg` | not part of the final comparison |
| StarGAN v2 | `make_2d_slices.py --split all --steps register,transform,normalize,inverse,slices`, then `train_stargan.py` | `predict_stargan.py --step 65200 --strategy best` |
| Diffusion | external, see `src/brainpred/methods/diffusion/README.md` | — |

On a SLURM cluster: `sbatch --export=ALL,MODEL=mgan slurm/train_gan.slurm` (edit the placeholders first).

MGAN, MGAN-seg and MPGAN use `tensorflow-addons` (TensorFlow ≤ 2.15, Keras 2). MPGAN needs the Models Genesis
weights (see [data.md](data.md)).

## 5. Postprocessing

The predictions go through the same SynthSR → skull stripping → normalization → FastSurfer chain as the inputs:

| Method | Command |
|---|---|
| MGAN | `postprocess.py --method MGAN_paper_16 --extra 46` |
| MPGAN | `postprocess.py --method MPGAN_paper_16 --extra 28` |
| MPGAN + seg. loss | `postprocess.py --method MPGAN_paper_16_seg_loss --extra 52 --source copy --steps copy,synthsr,skull_strip,normalize,segment,seg_to_patient,mask_ncrb`, then `--seg-input copy --seg-name SEG_96_RAW --steps segment,seg_to_patient,mask_ncrb` |
| StarGAN v2 | `postprocess.py --method StarGAN_V2_sagital --extra BEST --source reg --no-synthsr --steps register_back,skull_strip,normalize,segment,seg_to_patient,mask_ncrb` |
| Diffusion | `postprocess.py --method diffusion_model --extra RES --source reg --steps register_back,synthsr,skull_strip,normalize,segment,seg_to_patient,mask_ncrb` |
| Similar patches | `postprocess.py --method dfield_similar_patches --extra 96 10 --source copy --no-synthsr --seg-name SEG_96 --steps copy,skull_strip,normalize,segment,seg_to_patient,mask_ncrb` |
| Similar images | `postprocess.py --method dfield_similar_images --extra DICE_MEAN 100 --copy-as T1 --seg-input copy --seg-name SEG_96 --steps copy,segment,seg_to_patient` |

## 6. Evaluation

`evaluate.py` writes one CSV per metric (rows: subjects, columns: `--extra` values) to
`<results_dir>/<method>/statistics[_postprocessed]/`. For the thesis tables use
`--metrics ssim,ncc,psnr,bpf,dice`:

```bash
python scripts/evaluate.py --method dfield_mean --extra 10 100 200 300 620
python scripts/evaluate.py --method dfield_similar_images --extra SSIM_5 SSIM_10 SSIM_100 DICE_MEAN_5 DICE_MEAN_10 DICE_MEAN_100
python scripts/evaluate.py --method dfield_similar_images_registered --extra AFF_5 BSP_FAST_5
python scripts/evaluate.py --method dfield_similar_tissues --extra 3_10 3_100 96_10 96_100
python scripts/evaluate.py --method dfield_similar_patches --extra 96_10 --postprocessed --type-suffix _POST
python scripts/evaluate.py --method MGAN_paper_16 --extra 46 --postprocessed --type-suffix _POST
python scripts/evaluate.py --method MPGAN_paper_16 --extra 28 --postprocessed --type-suffix _POST
python scripts/evaluate.py --method MPGAN_paper_16_seg_loss --extra 52 --postprocessed --type-suffix _RAW --table-suffix _RAW
python scripts/evaluate.py --method StarGAN_V2_sagital --extra BEST --postprocessed --type-suffix _POST
python scripts/evaluate.py --method diffusion_model --extra RES --postprocessed --type-suffix _POST --split available
```

The "initial" baseline (HUNT3 compared with HUNT4) in the thesis tables comes from the longitudinal features
(`compute_features.py longitudinal_whole_brain` / `longitudinal_tissue`).

## 7. Figures and analysis

```bash
python scripts/make_figures.py --help      # compare | hunt | poster | method
```

The statistical analysis (demographics, longitudinal change, clustering, final comparison of all methods) is in
`notebooks/analysis/`. See `notebooks/README.md`.
