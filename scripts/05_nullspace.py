import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import VARIABLES
from src.nullspace import run_nullspace


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variables", nargs="+", default=VARIABLES)
    p.add_argument("--layers", nargs="+", type=int, required=True,
                   help="one layer per variable (same order), or a single layer for all")
    p.add_argument("--pooling", default="mean")
    p.add_argument("--hp", nargs="+", choices=["item1", "paper"], default=["paper"])
    p.add_argument("--max-iters", type=int, default=200, help="max probes per run")
    p.add_argument("--batch-size", type=int, default=128, help="0 = full batch")
    p.add_argument("--run-name", default="paper", help="Item 1 run to read item1 hparams from; output dir")
    p.add_argument("--stop-on", choices=["r2", "mae", "any"], default="any",
                   help="chance criterion: r2 only, mae only, or either (paper)")
    p.add_argument("--dir-remove", choices=["both", "sin", "cos"], default="both",
                   help="direction only: project out both probe rows, or only the sin / cos row")
    p.add_argument("--resume", action="store_true", help="skip runs whose _basis.npz already exists (finished)")
    p.add_argument("--out-subdir", default="nullspace", help="output folder under results/<run-name>/")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # One layer per variable, or broadcast a single layer to all
    layers = args.layers * len(args.variables) if len(args.layers) == 1 else args.layers
    if len(layers) != len(args.variables):
        sys.exit(f"--layers: got {len(args.layers)}, need 1 or {len(args.variables)}")

    # Run nullspace probing for each variable / hparam source
    summary = []
    for v, layer in zip(args.variables, layers):
        for hp in args.hp:
            summary.append(run_nullspace(
                v, args.pooling, layer, hp, run_name=args.run_name, max_iters=args.max_iters,
                batch_size=args.batch_size, seed=args.seed, device=args.device,
                stop_on=args.stop_on, dir_remove=args.dir_remove, resume=args.resume,
                out_subdir=args.out_subdir,
            ))

    # Summary table across runs
    print("\nSUMMARY")
    print(f"{'variable':<13}{'layer':>6}{'hp':>8}{'stop-on':>9}{'remove':>8}"
          f"{'base R²':>10}{'dims':>7}{'probes':>8}   stop")
    for s in summary:
        print(f"{s['variable']:<13}{s['layer']:>6}{s['hp']:>8}{s['stop_on']:>9}"
              f"{s['dir_remove']:>8}{s['baseline_r2']:>10.3f}{s['dim']:>7}{s['probes']:>8}   {s['stop']}")