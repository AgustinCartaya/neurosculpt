"""Compute per-patient evaluation metrics of a method's predictions against the real HUNT4 scans.

Examples (configurations used in the thesis):

    # MPGAN (postprocessed, checkpoint 28)
    python scripts/evaluate.py --method MPGAN_paper_16 --extra 28 --postprocessed --type-suffix _POST \\
        --metrics ssim,ncc,psnr,bpf,dice
    # Similar patches (postprocessed)
    python scripts/evaluate.py --method dfield_similar_patches --extra 96_10 --postprocessed --type-suffix _POST
    # MPGAN with segmentation loss: tables suffixed with _RAW / _POST
    python scripts/evaluate.py --method MPGAN_paper_16_seg_loss --extra 52 --postprocessed \\
        --type-suffix _RAW --table-suffix _RAW
    # Raw GAN outputs (no skull stripping / normalisation)
    python scripts/evaluate.py --method MGAN_paper_16 --extra 42 44 46 --raw --metrics ssim
    # Diffusion model: only the patients with results
    python scripts/evaluate.py --method diffusion_model --extra RES --postprocessed --type-suffix _POST \\
        --split available --metrics ssim,ncc,psnr,bpf,dice
"""
import argparse

from brainpred.evaluation import statistics
from brainpred.results import ResultsLayout


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, help="Method folder name in <results_dir>, e.g. MPGAN_paper_16.")
    parser.add_argument("--extra", nargs="+", required=True,
                        help="One or more result 'extra' strings (hyper-parameters joined with '_'), e.g. 46 or 96_10.")
    parser.add_argument("--postprocessed", action="store_true",
                        help="Read results_postprocessed/ and write statistics_postprocessed/.")
    parser.add_argument("--type-suffix", default="", help="Image type suffix, e.g. _POST or _RAW.")
    parser.add_argument("--table-suffix", default="", help="Suffix appended to the CSV table names.")
    parser.add_argument("--metrics", default="ncc,psnr",
                        help=f"Comma-separated subset of {','.join(statistics.METRICS)}.")
    parser.add_argument("--split", default="validation", choices=["validation", "train", "available"],
                        help="'available' = patients that have a result folder (used for the diffusion model).")
    parser.add_argument("--raw", action="store_true",
                        help="Compare un-skull-stripped, un-normalised images (ssim/ncc/psnr only).")
    parser.add_argument("--summary", action="store_true", help="Print mean / std (x100) of each table.")
    return parser.parse_args()


def main():
    args = parse_args()
    layout = ResultsLayout(args.method)
    ids = statistics.get_ids(args.split, layout, args.postprocessed)
    metrics_list = [m.strip() for m in args.metrics.split(",") if m.strip()]

    tables = statistics.compute_statistics(layout, ids, args.extra, metrics_list,
                                           postprocessed=args.postprocessed, type_suffix=args.type_suffix,
                                           table_suffix=args.table_suffix, raw=args.raw)
    if args.summary:
        summary = statistics.summarize(tables)
        print("mean\n", summary["mean"], "\nstd\n", summary["std"])


if __name__ == "__main__":
    main()
