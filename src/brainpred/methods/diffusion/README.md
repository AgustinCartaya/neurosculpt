# Diffusion model

The diffusion-model predictions of the thesis were produced **outside this repository** (2D latent
diffusion model trained by a collaborator on axial slices); there is no training or inference code here.
`OLD/test_code/MONAI/LDM_2D` was only the MONAI tutorial and was not ported.

To evaluate such predictions with this package:

1. Put the predicted 3D volume of each patient, in the HUNT3 *PREP_MNI* space used for the 2D slices, at

       <results_dir>/diffusion_model/results/<id>/<id>_0_T1_RES.nii.gz

   (`ResultsLayout("diffusion_model").get_result_image_path_name(id, "T1", ("RES",))`).
   If you have per-slice PNGs (`<slices_dir>/<id>/<id>_1_T1_<layer>.png`, layers 34-156, 160x224), build
   the volumes with

       python -c "from brainpred.methods.diffusion import reconstruction as r; r.save_3D_images('<slices_dir>')"

2. Run the postprocessing (registration back to the patient space, SynthSR, skull stripping,
   normalisation, FastSurfer segmentation, no-cerebellum mask), which writes
   `results_postprocessed/<id>/<id>_0_{T1,SEG_96,BMASK_NCRB}_POST_RES.nii.gz`.

3. Evaluate and visualise (only the patients that have results):

       python scripts/evaluate.py --method diffusion_model --extra RES --postprocessed --type-suffix _POST \
           --split available --metrics ssim,ncc,psnr,bpf,dice
       python scripts/make_figures.py method --method diffusion_model --kind seg --case <id> --extras RES \
           --postprocessed --include-hunt3

In the thesis 83 patients were evaluated: all validation patients with a prediction, completed with
other patients (`reconstruction.select_evaluation_ids`).
