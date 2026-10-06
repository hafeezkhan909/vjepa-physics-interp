import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UNIT = {"direction": "deg", "speed": "m/s", "acceleration": "m/s$^2$"}
SYM = {"direction": "°", "speed": "", "acceleration": ""}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--run-name", default="paper")
    p.add_argument("--variables", nargs="*", default=["direction", "speed", "acceleration"])
    p.add_argument("--pooling", default="mean")
    p.add_argument("--layers", nargs="*", type=int, default=None, help="default: all found")
    p.add_argument("--eval", nargs="*", default=["ridge"], help="eval-probe tags to include (ridge, sgd0, ...)")
    p.add_argument("--mode", choices=["target", "avg"], default="target",
                   help="target = the single target the run was made with; avg = mean over all targets")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run = args.results_dir / args.run_name
    out_dir = run / "figs" / "steering"
    out_dir.mkdir(parents=True, exist_ok=True)

    for v in args.variables:
        for f in sorted((run / "steering").glob(f"{v}_{args.pooling}_L*_eval-*.csv")):
            # Filename = var_pool_Lxx_eval-xxx
            parts = f.stem.split("_")
            layer, ev = int(parts[2][1:]), parts[3].replace("eval-", "")
            if ev not in args.eval or (args.layers and layer not in args.layers):
                continue
            df = pd.read_csv(f)

            # Pick the run's own target, or average over all targets
            if args.mode == "target":
                t0 = df["target"].iloc[0]  # first target in each K block = the run's target
                d = df[df["target"] == t0].sort_values("K")
                label = f"target={t0:g}{SYM[v]}"
            else:
                if df["target"].nunique() < 2:
                    print(f"skip {f.name}: run without --avg-targets")
                    continue
                d = df.groupby("K", as_index=False)[["mae_to_target", "mae_to_truth"]].mean()
                label = f"avg over {df['target'].nunique()} targets"

            # MAE to target and to true label vs K
            fig, ax = plt.subplots(figsize=(6.5, 4.3))
            ax.plot(d["K"], d["mae_to_target"], "-o", ms=3, lw=2, color="#d62728", label="MAE to target")
            ax.plot(d["K"], d["mae_to_truth"], "-o", ms=3, lw=2, color="#1f77b4", label="MAE to true label")
            ax.set_xlabel("number of probes used to steer (K)")
            ax.set_ylabel(f"MAE ({UNIT[v]}), held-out eval probe")
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=9)
            ax.set_title(f"{v} | Layer {layer} | {label} | eval probe R²={df['eval_r2'].iloc[0]:.2f}", fontsize=10)
            fig.tight_layout()
            out = out_dir / f"{f.stem}_{args.mode}.png"
            fig.savefig(out, dpi=200)
            plt.close(fig)
            print("saved", out)