#!/usr/bin/env python
"""Produce plots/errors from an already-trained checkpoint, without retraining.

Run with `pip install -e .` done once, from *any* directory - the example name always
resolves against <repo_root>/examples/, regardless of the caller's cwd or how this script
itself was invoked:

    python rstnets/restore.py --case phill
    python rstnets/restore.py --case phill --checkpoint-dir examples/phill/results/Model-adam
    cd rstnets && python restore.py --case phill          # same result, run from inside rstnets/
    python rstnets/restore.py phill                        # --case is optional; a bare positional works too

Defaults to loading <example_dir>/results/Model-lbfgs/ (where `train.py` saves its final
checkpoint) and writes plots/errors back into <example_dir>/results/ alongside it.
"""
import argparse
import os
import sys

if "--no-gpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from rstnets.config import CaseConfig, RunConfig, resolve_example


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case_positional", nargs="?", default=None, metavar="case",
                         help="Example name under examples/ (e.g. 'phill') or a path to a case directory")
    parser.add_argument("--case", dest="case_flag", default=None, metavar="CASE",
                         help="Same as the positional argument; takes precedence if both are given")
    parser.add_argument("--output-dir", default=None,
                         help="Defaults to <example_dir>/results")
    parser.add_argument("--checkpoint-dir", default=None,
                         help="Defaults to <output_dir>/Model-lbfgs")
    parser.add_argument("--backend", choices=["tensorflow", "torch"], default=None,
                         help="Overrides the case config's backend (default: use the case config's own choice)")
    parser.add_argument("--precision", choices=["float32", "float64"], default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU (sets CUDA_VISIBLE_DEVICES='')")
    parser.add_argument("--dry-run", action="store_true",
                         help="Validate case/data without restoring")
    args = parser.parse_args(argv)

    case = args.case_flag or args.case_positional
    if case is None:
        parser.error("a case name/path is required, e.g. 'python rstnets/restore.py --case phill'")
    args.case = case
    return args


def main(argv=None):
    args = parse_args(argv)
    example_dir = resolve_example(args.case)
    output_dir = args.output_dir or str(example_dir / "results")

    run_config = RunConfig(
        data_path=str(example_dir), output_dir=output_dir, backend=args.backend,
        checkpoint_dir=args.checkpoint_dir, precision=args.precision, seed=args.seed, gpu=not args.no_gpu,
    )
    case_config = CaseConfig.load(example_dir)

    from rstnets.training import restore_case  # deferred: imports tensorflow or torch
    restore_case(run_config, case_config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
