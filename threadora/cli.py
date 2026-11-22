"""Command line: python -m threadora.cli IMAGE --width 5 --height 5 --units in --name Cecilia ..."""
import argparse
import json

from .pipeline import run_order


def main(argv=None):
    ap = argparse.ArgumentParser(description="THREADORA V10 cross-stitch engine")
    ap.add_argument("image")
    ap.add_argument("--width", type=float, required=True)
    ap.add_argument("--height", type=float, required=True)
    ap.add_argument("--units", default="in", choices=["in", "cm"])
    ap.add_argument("--name", default="")
    ap.add_argument("--phrase", default="")
    ap.add_argument("--year", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--display-fabric", default=None,
                    help="use this fabric (grid fitted to it inside the requested size; size never changes)")
    ap.add_argument("--no-tests", action="store_true")
    ap.add_argument("--review-copy", action="store_true",
                    help="on a size-fidelity conflict, still build the best PDF marked REVIEW COPY - NOT VERIFIED")
    a = ap.parse_args(argv)
    res = run_order(a.image, a.width, a.height, a.units,
                    {"phrase": a.phrase, "name": a.name, "year": a.year}, a.out,
                    display_fabric=a.display_fabric, run_tests=not a.no_tests, review_copy=a.review_copy)
    print(json.dumps({k: res.get(k) for k in ("status", "blocker", "pdf", "pdf_pages", "dmc_colors",
                                               "finished_in", "display_fabric", "review_reasons", "tests")}, indent=2, default=str))


if __name__ == "__main__":
    main()
