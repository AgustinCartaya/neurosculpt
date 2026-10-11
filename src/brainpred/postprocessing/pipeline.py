"""Generic postprocessing of the predicted HUNT4 images of any method (replaces every ``approaches/*/postprocess_images.py``).

Every step reads and writes ``layout.get_result_image_path_name(id, <TYPE>, extra_info, postprocessed=True)``
(the prediction itself is the non-postprocessed ``T1``). Steps, in pipeline order:

=================  =============================================  ==========================
step               input                                          output
=================  =============================================  ==========================
copy               T1 (raw prediction)                            <copy_as> (default T1_RAW)
register_back      T1 (raw prediction, PREP_MNI space)            T1_REG
synthsr            source image                                   T1_SYNTHSR
skull_strip        intensity image                                BMASK
normalize          intensity image + BMASK                        T1_POST
segment            seg input (FastSurfer, GPU machine)            <seg_name>.mgz
seg_to_patient     <seg_name>.mgz + seg input                     <seg_name> (.nii.gz)
mask_ncrb          BMASK + <seg_name>                             <seg_name with SEG_96 -> BMASK_NCRB>
=================  =============================================  ==========================

* source image: ``raw`` = T1 prediction, ``copy`` = the copied image, ``reg`` = T1_REG;
* intensity image: T1_SYNTHSR if ``use_synthsr`` else the source image;
* seg input: ``post`` = T1_POST, ``copy`` = the copied image.

``segment`` needs FastSurfer (docker); the other external steps need FreeSurfer. Historically
they ran on different machines, one step at a time.
"""
import os
import shutil

from tqdm import tqdm

from brainpred import paths
from brainpred.postprocessing import steps as post
from brainpred.utils import io

STEPS = ["copy", "register_back", "synthsr", "skull_strip", "normalize", "segment", "seg_to_patient", "mask_ncrb"]


def postprocess_patient(layout, id, extra_info, steps, source="raw", copy_as="T1_RAW", use_synthsr=True,
                        seg_input="post", seg_name="SEG_96_POST", gpu=True):
    """Run ``steps`` (subset of :data:`STEPS`, executed in pipeline order) for one patient."""
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise ValueError(f"Unknown postprocessing steps: {sorted(unknown)}")

    def post_path(type_img):
        return layout.get_result_image_path_name(id, type_img, extra_info, postprocessed=True)

    raw_path = layout.get_result_image_path_name(id, "T1", extra_info)
    source_path = {"raw": raw_path, "copy": post_path(copy_as), "reg": post_path("T1_REG")}[source]
    intensity_path = post_path("T1_SYNTHSR") if use_synthsr else source_path
    bmask_path = post_path("BMASK")
    t1_post_path = post_path("T1_POST")
    seg_input_path = {"post": t1_post_path, "copy": post_path(copy_as)}[seg_input]
    seg_path = post_path(seg_name)
    bmask_ncrb_path = post_path(seg_name.replace("SEG_96", "BMASK_NCRB"))

    io.verify_folder(layout.get_result_patient_folder(id, postprocessed=True), create=True)

    if "copy" in steps:
        shutil.copy(raw_path, post_path(copy_as))
    if "register_back" in steps:
        post.register_image_back(id, raw_path, post_path("T1_REG"))
    if "synthsr" in steps:
        post.save_synthSR(source_path, post_path("T1_SYNTHSR"))
    if "skull_strip" in steps:
        post.save_skull_stripping_freesurfer(intensity_path, bmask_path)
    if "normalize" in steps:
        post.save_normalized_image(intensity_path, bmask_path, t1_post_path)
    if "segment" in steps:
        post.segment_freesurfer(id, seg_input_path, seg_path, GPU=gpu)
    if "seg_to_patient" in steps:
        post.transform_segmentation_to_patient_space(id, seg_path, seg_input_path)
    if "mask_ncrb" in steps:
        post.create_mask_no_cerebellum(bmask_path, seg_path, bmask_ncrb_path)


def postprocess(layout, ids, extra_info, steps, **kwargs):
    for id in tqdm(ids):
        postprocess_patient(layout, id, extra_info, steps, **kwargs)


def segment_single_image(sid, img_path_name, seg_path_name, steps=("segment", "seg_to_patient"), gpu=True):
    """Segment one image outside the results layout (e.g. the MNI template, former ``approaches/0MNI_Segmentation``)."""
    if "segment" in steps:
        post.segment_freesurfer(sid, img_path_name, seg_path_name, GPU=gpu)
    if "seg_to_patient" in steps:
        post.transform_segmentation_to_patient_space(sid, seg_path_name, img_path_name)


# ---- Early MGAN results named "<id>_CHECK_<checkpoint>.nii.gz" (former postprocessing.py::do_for_MGAN_PAPER)
MGAN_PAPER_STEPS = ["synthsr", "skull_strip", "hist_match"]


def postprocess_mgan_paper_checkpoints(path_in, ids, checkpoints, steps):
    """SynthSR, skull stripping and histogram matching to the patient's HUNT3 ``T1_PREP``, written to ``<path_in>/postprocessed``."""
    path_out = f"{path_in}/postprocessed"
    io.verify_folder(path_out, create=True)
    org_name = "{0}_CHECK_{1}.nii.gz"
    prep_name = "{0}_CHECK_{1}{2}.nii.gz"
    bmask_name = "{0}_CHECK_{1}_BMASK.nii.gz"

    for id in ids:
        for check in checkpoints:
            synthsr_path = os.path.join(path_out, prep_name.format(id, check, "_SYNTHSR"))
            bmask_path = os.path.join(path_out, bmask_name.format(id, check))
            if "synthsr" in steps:
                print(f"Step 1: SynthSR for {id} {check}")
                post.save_synthSR(os.path.join(path_in, org_name.format(id, check)), synthsr_path)
            if "skull_strip" in steps:
                print(f"Step 2: Skull stripping for {id} {check}")
                post.save_skull_stripping_freesurfer(synthsr_path, bmask_path)
            if "hist_match" in steps:
                print(f"Step 3: Histogram matching for {id} {check}")
                post.save_histogram_match(paths.get_preprocessed_image_path_name(3, id), synthsr_path,
                                          os.path.join(path_out, prep_name.format(id, check, "_POST")), bmask_path)
