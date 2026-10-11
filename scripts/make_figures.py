"""Qualitative result figures (comparison of all methods, poster figures, per-method viewers).

Patients can be given as an id or as LOW / MEDIUM / HIGH (example validation patients of the thesis).
Figures are interactive; pass --save to also write them (DPI 1000) to <results_dir>/figures/.

Examples:
    # comparison of the GAN family, axial slice 110
    python scripts/make_figures.py compare --kind results --case LOW --family GAN --plane 2 --layer 110
    # error maps and important tissues of the best methods (as saved for the report)
    python scripts/make_figures.py compare --kind dif --case HIGH --family BEST --layer 103 --save
    python scripts/make_figures.py compare --kind seg_important --case HIGH --family BEST --layer 103 --save
    python scripts/make_figures.py compare --kind seg_important --case MNI --family MNI --plane 1 --layer 69 --save
    # poster: deformation field predicted by the similar-tissues method, only arrows
    python scripts/make_figures.py poster --kind df --case LOW --family SIM_TIS --nb-tissues 3 --only-df \\
        --extra-info sim_tissue_df --save
    # per-method viewer (MPGAN, raw checkpoints skull-stripped with the postprocessed segmentation of checkpoint 28)
    python scripts/make_figures.py method --method MPGAN_paper_16 --kind results --case 0 --extras 9 25 46 \\
        --seg-source 28 _POST 1 --normalize-if-not-stripped
"""
import argparse

from brainpred import splits
from brainpred.evaluation import figures


def _patient(case):
    """LOW/MEDIUM/HIGH, an integer index into the validation ids, or an id."""
    if case in figures.CASES or not case.isdigit() or len(case) == 5:
        return figures.get_case_id(case)
    return splits.get_validation_ids()[int(case)]


def _family(values):
    return values[0] if len(values) == 1 and values[0] in figures.FAMILIES else values


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, plane=2, layer=110):
        p.add_argument("--case", default="LOW", help="LOW/MEDIUM/HIGH, an index into the validation ids, or a patient id.")
        p.add_argument("--plane", type=int, default=plane, help="Moving plane (0 sagittal, 1 coronal, 2 axial).")
        p.add_argument("--layer", type=int, default=layer, help="Initial / shown slice.")
        p.add_argument("--save", action="store_true")

    p = sub.add_parser("compare", help="Comparison of all methods (approaches/visualize_results.py).")
    common(p)
    p.add_argument("--kind", required=True, choices=["results", "dif", "seg_dif", "seg", "seg_important"])
    p.add_argument("--family", nargs="+", default=["DF"], help=f"One of {list(figures.FAMILIES)} or a list of entries "
                                                                f"({['HUNT3', 'HUNT4', 'REG', 'MNI'] + list(figures.RESULT_SOURCES)}).")
    p.add_argument("--nb-tissues", type=int, default=96, choices=[3, 36, 96])
    p.add_argument("--see-tissues", type=int, nargs="+", default=None, help="Labels to keep (default: all / IMPORTANT_TISSUES).")

    p = sub.add_parser("hunt", help="HUNT3 vs HUNT4 images, or images and segmentations, of several patients.")
    p.add_argument("--kind", required=True, choices=["hunt3_hunt4", "hunt_seg"])
    p.add_argument("--cases", nargs="+", default=["LOW", "MEDIUM", "HIGH"])
    p.add_argument("--hunt", type=int, default=4)
    p.add_argument("--nb-tissues", type=int, default=3)
    p.add_argument("--plane", type=int, default=2)
    p.add_argument("--layer", type=int, default=110)

    p = sub.add_parser("poster", help="Single-slice / deformation-field figures (approaches/visualize_results_poster.py).")
    common(p, layer=113)
    p.add_argument("--kind", required=True, choices=["df", "image", "tissue"])
    p.add_argument("--family", nargs="+", default=["REG"])
    p.add_argument("--nb-tissues", type=int, default=3)
    p.add_argument("--see-tissues", type=int, nargs="+", default=None)
    p.add_argument("--extra-info", default="", help="Tag added to the file name; sim_tissue[_df] also draws tissue borders.")
    p.add_argument("--only-df", action="store_true", help="Draw only the arrows (transparent background).")
    p.add_argument("--df-layer", type=int, default=113, help="Axial slice of the deformation field.")

    p = sub.add_parser("method", help="Per-method viewers (approaches/<method>/visualize_results.py).")
    p.add_argument("--method", required=True)
    p.add_argument("--kind", required=True, choices=["results", "diff", "seg"])
    p.add_argument("--case", default="0")
    p.add_argument("--extras", nargs="+", required=True)
    p.add_argument("--postprocessed", action="store_true")
    p.add_argument("--type-suffix", default=None, help="Default: _POST if --postprocessed else ''.")
    p.add_argument("--no-skull-stripping", action="store_true")
    p.add_argument("--normalize-if-not-stripped", action="store_true", help="GAN viewers: normalise when not skull-stripping.")
    p.add_argument("--seg-source", nargs=3, metavar=("EXTRA", "TYPE_SUFFIX", "POSTPROCESSED"), default=None,
                   help="Segmentation used to skull-strip, e.g. 28 _POST 1.")
    p.add_argument("--tissue", default=None, help="CSF/GM/WM or a SEG_96 label (similar-patches viewer).")
    p.add_argument("--name-prefix", default="", help="Prefix of the result titles (RES_ for dfield_mean).")
    p.add_argument("--hunt3-first", action="store_true", help="diff: diffusion-model layout.")
    p.add_argument("--include-hunt3", action="store_true", help="seg: also show HUNT3 (diffusion-model layout).")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "compare":
        id = "MNI" if args.case == "MNI" else _patient(args.case)
        family = _family(args.family)
        kw = dict(family=family, mooving_plane=args.plane, current_plane=args.layer, save=args.save)
        if args.kind == "results":
            figures.visualize_final_results(id, **kw)
        elif args.kind == "dif":
            figures.visualize_final_results_dif(id, **kw)
        elif args.kind == "seg_dif":
            figures.visualize_final_results_seg_dif(id, nb_tissues=args.nb_tissues, see_tissues=args.see_tissues, **kw)
        elif args.kind == "seg":
            figures.visualize_final_results_seg(id, nb_tissues=args.nb_tissues, see_tissues=args.see_tissues, **kw)
        else:
            figures.visualize_final_results_seg_important_tissues(id, see_tissues=args.see_tissues or figures.IMPORTANT_TISSUES, **kw)

    elif args.command == "hunt":
        ids = [_patient(c) for c in args.cases]
        if args.kind == "hunt3_hunt4":
            figures.visualize_HUNT3_HUNT4_images(ids, mooving_plane=args.plane, current_plane=args.layer)
        else:
            figures.visualize_HUNT_SEG_images(ids, hunt=args.hunt, nb_tissues=args.nb_tissues, mooving_plane=args.plane,
                                              current_plane=args.layer)

    elif args.command == "poster":
        id = _patient(args.case)
        kw = dict(family=args.family, extra_info=args.extra_info, mooving_plane=args.plane, current_plane=args.layer, save=args.save)
        if args.kind == "df":
            figures.visualize_final_df(id, nb_tissues=args.nb_tissues, see_tissues=args.see_tissues, only_df=args.only_df,
                                       layer=args.df_layer, **kw)
        elif args.kind == "image":
            figures.visualize_image_alone(id, **kw)
        else:
            figures.visualize_tissue_alone(id, args.see_tissues, nb_tissues=args.nb_tissues, **kw)

    else:
        id = _patient(args.case)
        tissue = int(args.tissue) if args.tissue and args.tissue.isdigit() else args.tissue
        seg_source = None
        if args.seg_source:
            extra, suffix, post = args.seg_source
            seg_source = (extra, suffix, post.lower() in ("1", "true", "yes"))
        if args.kind == "results":
            figures.visualize_results(args.method, id, args.extras, args.postprocessed, args.type_suffix,
                                      skull_stripping=not args.no_skull_stripping, seg_source=seg_source,
                                      normalize_if_not_stripped=args.normalize_if_not_stripped, name_prefix=args.name_prefix)
        elif args.kind == "diff":
            figures.visualize_results_diff(args.method, id, args.extras, args.postprocessed, args.type_suffix, seg_source,
                                           tissue=tissue, name_prefix=args.name_prefix, hunt3_first=args.hunt3_first)
        else:
            figures.visualize_results_and_seg(args.method, id, args.extras, args.postprocessed, args.type_suffix, tissue=tissue,
                                              name_prefix=args.name_prefix, include_hunt3=args.include_hunt3)


if __name__ == "__main__":
    main()
