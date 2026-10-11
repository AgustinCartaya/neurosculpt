"""Compute the patient feature tables (written to <work_dir>/features/<kind>/...)."""
import argparse

from brainpred import splits
from brainpred.features.approaches import common as approaches_common
from brainpred.features.approaches import tissue_based as approaches_tissue_based
from brainpred.features.approaches import whole_brain_based as approaches_whole_brain_based
from brainpred.features.cross_sectional import tissue_based as cross_sectional_tissue_based
from brainpred.features.cross_sectional import whole_brain as cross_sectional_whole_brain
from brainpred.features.inter_patient import patch_based as inter_patient_patch_based
from brainpred.features.inter_patient import tissue_based as inter_patient_tissue_based
from brainpred.features.inter_patient import whole_brain_based as inter_patient_whole_brain_based
from brainpred.features.longitudinal import tissue_based as longitudinal_tissue_based
from brainpred.features.longitudinal import whole_brain_based as longitudinal_whole_brain_based
from brainpred.features.registration import tissue_based as registration_tissue_based
from brainpred.features.registration import whole_brain_based as registration_whole_brain_based

KINDS_HELP = """\
kinds (and what needs them):
  cross_sectional_whole_brain  BPF at HUNT3/HUNT4                         (StarGAN v2 reference selection)
  cross_sectional_tissue       tissue volumes                              (needed by inter_patient_tissue)
  longitudinal_whole_brain     BPF change + HUNT3 vs HUNT4 NCC/SSIM/PSNR   (analysis notebooks)
  longitudinal_tissue          per-tissue Dice/volume change + dfield stats (needs longitudinal registrations)
  registration_whole_brain     same as longitudinal, registered HUNT3 vs HUNT4
  registration_tissue          per-tissue Dice/volume change, registered HUNT3 vs HUNT4
  inter_patient_whole_brain    pairwise NCC/SSIM                            (dfield similar_images --sim-metrics ssim)
  inter_patient_tissue         pairwise per-tissue Dice / volume difference (dfield similar_images(_registered), similar_tissues, GAN postprocessing)
  inter_patient_patch          validation-vs-reference patch Dice          (dfield similar_patches)
  approaches_whole_brain       BPF / SSIM / NCC / PSNR of the predictions   (final macro study)
  approaches_tissue            tissue volume / Dice of the predictions      (final macro study)
"""


def add_nb_tissues(parser, default):
    parser.add_argument("--nb-tissues", type=int, nargs="+", default=default, choices=[3, 36, 96])


def add_approach(parser):
    parser.add_argument("--approach", help="method results folder (default: every configuration of the final comparison)")
    parser.add_argument("--extra-info", nargs="+", help="extra_info of the result files, e.g. 96 10")


def get_approaches(args):
    if args.approach is None:
        return list(approaches_common.FINAL_APPROACHES.items())
    extra_info = args.extra_info if args.extra_info is not None else approaches_common.FINAL_APPROACHES[args.approach]
    return [(args.approach, extra_info)]


def run(args):
    kind = args.kind
    if kind == "cross_sectional_whole_brain":
        cross_sectional_whole_brain.compute_patient_metrics_for_all()
    elif kind == "cross_sectional_tissue":
        for nb_tissues in args.nb_tissues:
            cross_sectional_tissue_based.compute_patient_metrics_for_all(nb_tissues)
    elif kind == "longitudinal_whole_brain":
        longitudinal_whole_brain_based.compute_patient_metrics_for_all()
    elif kind == "longitudinal_tissue":
        longitudinal_tissue_based.compute_patient_metrics_for_all(args.nb_tissues)
    elif kind == "registration_whole_brain":
        registration_whole_brain_based.compute_patient_metrics_for_all()
    elif kind == "registration_tissue":
        ids = splits.get_validation_ids() if args.split == "validation" else splits.get_matching_ids()
        registration_tissue_based.compute_patient_metrics_for_all(ids, args.nb_tissues)
    elif kind == "inter_patient_whole_brain":
        inter_patient_whole_brain_based.create_similarity_matrix(hunt=args.hunt)
    elif kind == "inter_patient_tissue":
        exclude = args.exclude_tissues or None
        for nb_tissues in args.nb_tissues:
            inter_patient_tissue_based.create_similarity_matrix(nb_tissues, tissues_ids=exclude, excluded_tissues=exclude is not None, hunt=args.hunt)
    elif kind == "inter_patient_patch":
        ids_val = splits.get_validation_ids()
        excluded = splits.get_excluded_patients() + ids_val
        ids_train = [id for id in splits.get_train_ids() if id not in excluded]
        for nb_tissues in args.nb_tissues:
            inter_patient_patch_based.create_and_save_similarity_patches_tables(ids_val, ids_train, nb_tissues)
    elif kind == "approaches_whole_brain":
        for approach, extra_info in get_approaches(args):
            approaches_whole_brain_based.compute_patient_metrics_for_all(approach, extra_info)
    elif kind == "approaches_tissue":
        for approach, extra_info in get_approaches(args):
            approaches_tissue_based.compute_patient_metrics_for_all(approach, extra_info, args.nb_tissues)


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=KINDS_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="kind", required=True, metavar="kind")

    subparsers.add_parser("cross_sectional_whole_brain")
    add_nb_tissues(subparsers.add_parser("cross_sectional_tissue"), [3, 36, 96])
    subparsers.add_parser("longitudinal_whole_brain")
    add_nb_tissues(subparsers.add_parser("longitudinal_tissue"), [3, 36, 96])
    subparsers.add_parser("registration_whole_brain")
    p = subparsers.add_parser("registration_tissue")
    add_nb_tissues(p, [96])
    p.add_argument("--split", choices=["validation", "all"], default="validation")

    p = subparsers.add_parser("inter_patient_whole_brain")
    p.add_argument("--hunt", type=int, choices=splits.HUNTS, default=3)
    p = subparsers.add_parser("inter_patient_tissue", help="needs cross_sectional_tissue for the same --nb-tissues")
    add_nb_tissues(p, [3, 96])
    p.add_argument("--hunt", type=int, choices=splits.HUNTS, default=3)
    p.add_argument("--exclude-tissues", type=int, nargs="*", default=[0], help="tissues left out (default: background)")
    add_nb_tissues(subparsers.add_parser("inter_patient_patch"), [96])

    add_approach(subparsers.add_parser("approaches_whole_brain"))
    p = subparsers.add_parser("approaches_tissue")
    add_approach(p)
    add_nb_tissues(p, [3, 36, 96])

    run(parser.parse_args())


if __name__ == "__main__":
    main()
