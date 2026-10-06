import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.steering import run_steering


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variables", nargs="+", default=["direction"], choices=["direction", "speed", "acceleration"])
    p.add_argument("--layers", nargs="+", type=int, required=True)
    p.add_argument("--target", type=float, default=None,
                   help="override target (default: 90° for direction; label just above the mean for speed/acceleration)")
    p.add_argument("--avg-targets", action="store_true", help="also steer to every label value and average")
    p.add_argument("--k-step", type=int, default=1, help="sweep K = 1, 1+step, ... (K_max always included)")
    p.add_argument("--eval-probe", nargs="+", choices=["sgd", "ridge"], default=["ridge"],
                   help="held-out eval probe: ridge (closed form, default) and/or sgd (paper hp)")
    p.add_argument("--eval-seed", type=int, default=0, help="init seed of the sgd eval probe (steering probes use 0)")
    p.add_argument("--pooling", default="mean")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--run-name", default="paper")
    p.add_argument("--steer-dir", default="steer_orth", help="folder with saved steering probes (W, b) under results/<run-name>/")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run = args.results_dir / args.run_name
    out_dir = run / "steering"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Steer for each variable / layer / eval probe, sweeping K
    for v in args.variables:
        for layer in args.layers:
            for ev in args.eval_probe:
                rows = run_steering(
                    v, args.pooling, layer, run / args.steer_dir,
                    args.target, args.avg_targets, k_step=args.k_step, seed=args.seed, device=args.device,
                    eval_probe=ev, eval_seed=args.eval_seed,
                )

                # Save one CSV per run
                tag = f"eval-sgd{args.eval_seed}" if ev == "sgd" else "eval-ridge"
                out = out_dir / f"{v}_{args.pooling}_L{layer:02d}_{tag}.csv"
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"saved {out}")