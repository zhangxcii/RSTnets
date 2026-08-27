#!/usr/bin/env python
"""Train a case from scratch (Adam then L-BFGS).

Run with `pip install -e .` done once, from *any* directory - the example name always
resolves against <repo_root>/examples/, regardless of the caller's cwd or how this script
itself was invoked:

    python rstnets/train.py --case phill
    python rstnets/train.py --case my_flow             # examples/my_flow/, e.g. copied from examples/template/
    cd rstnets && python train.py --case phill         # same result, run from inside rstnets/
    python /abs/path/to/RSTnets/rstnets/train.py --case phill   # same result, run from anywhere else
    python rstnets/train.py --case ./some/other/case   # a path (contains "/") is used directly, not looked up in examples/
    python rstnets/train.py phill                      # --case is optional; a bare positional works too

Checkpoints, loss history, and plots/errors are written to <example_dir>/results/ by default.
"""
import argparse
import os
import sys

# Set before any deferred `rstnets`/tensorflow import that --no-gpu below might trigger.
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
    parser.add_argument("--n-iters-adam", dest="n_iters_adam", type=int, default=None)
    parser.add_argument("--n-lbfgs", dest="n_lbfgs", type=int, default=None)
    parser.add_argument("--print-interval", dest="print_interval", type=int, default=None,
                         help="Overrides the case config's training_defaults.print_interval")
    parser.add_argument("--backend", choices=["tensorflow", "torch"], default=None,
                         help="Overrides the case config's backend (default: use the case config's own choice)")
    parser.add_argument("--precision", choices=["float32", "float64"], default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU (sets CUDA_VISIBLE_DEVICES='')")
    parser.add_argument("--dry-run", action="store_true",
                         help="Validate case/data/model without training")
    args = parser.parse_args(argv)

    case = args.case_flag or args.case_positional
    if case is None:
        parser.error("a case name/path is required, e.g. 'python rstnets/train.py --case phill'")
    args.case = case
    return args


def main(argv=None):
    args = parse_args(argv)
    example_dir = resolve_example(args.case)
    output_dir = args.output_dir or str(example_dir / "results")

    run_config = RunConfig(
        data_path=str(example_dir), output_dir=output_dir, backend=args.backend,
        precision=args.precision, n_iters_adam=args.n_iters_adam, n_lbfgs=args.n_lbfgs,
        print_interval=args.print_interval, seed=args.seed, gpu=not args.no_gpu,
    )
    case_config = CaseConfig.load(example_dir)

    from rstnets.training import train_case  # deferred: imports tensorflow or torch
    train_case(run_config, case_config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
