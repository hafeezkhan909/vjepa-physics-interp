import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
N_BLOCKS = 24
COLORS = {"speed": "#1f77b4", "direction": "#d62728", "acceleration": "#2ca02c"}
UNITS = {"speed": "m/s", "direction": "deg", "acceleration": "m/s$^2$"}


def make_figure(data, m, source, args, out):
    # source = 'cv' (mean +- std band) or 'test' (single held-out score per layer)
    # R2: all variables on one axis; MAE: one subplot per variable (different units)
    if m == "r2":
        fig, ax = plt.subplots(figsize=(7, 4.8))
        axes = {v: ax for v in data}
    else:
        fig, axs = plt.subplots(1, len(data), figsize=(5 * len(data), 4.2), squeeze=False)
        axes = dict(zip(data, axs[0]))

    # Plot each variable's curve
    for v, df in data.items():
        ax = axes[v]
        x = df["layer"] / N_BLOCKS if args.x_fraction else df["layer"]
        if source == "cv":
            mean, std = df[f"cv_{m}_mean"], df[f"cv_{m}_std"]
            ax.plot(x, mean, color=COLORS[v], lw=2, label=f"{v}")
            ax.fill_between(x, mean - std, mean + std, color=COLORS[v], alpha=0.2)
        else:
            ax.plot(x, df[f"test_{m}"], color=COLORS[v], lw=2, marker="o", ms=3, label=f"{v} (test)")
        if m == "mae":
            ax.set_title(v)
            ax.set_ylabel(f"{'test ' if source == 'test' else ''}MAE ({UNITS[v]})")

    # Axis styling + optional marked layer / shaded span
    scale = (lambda L: L / N_BLOCKS) if args.x_fraction else (lambda L: L)
    for ax in set(axes.values()):
        if args.mark_layer is not None:
            if args.mark_span > 0:
                lo, hi = args.mark_layer - args.mark_span, args.mark_layer + args.mark_span
                ax.axvspan(scale(lo), scale(hi), color="lightgray", alpha=0.4, lw=0, zorder=0,
                           label=f"layers {lo}–{hi}")
            ax.axvline(scale(args.mark_layer), color="gray", ls="--", lw=1.2, label=f"layer {args.mark_layer}")
        ax.set_xlabel("Layer fraction" if args.x_fraction else "Layer")
        ax.grid(alpha=0.25)
        if m == "r2":
            ax.set_ylim(args.ymin, args.ymax)
            ax.set_yticks(np.arange(args.ymin, args.ymax + 1e-6, 0.2))
        ax.legend(fontsize=9)
    if m == "r2":
        ax.set_ylabel("R$^2$ (Cross Validation, mean $\\pm$ std over folds)" if source == "cv" else "R$^2$ (held-out test)")

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved", out)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--run-name", default="paper")
    p.add_argument("--variables", nargs="*", default=["speed", "direction", "acceleration"])
    p.add_argument("--poolings", nargs="*", default=["mean", "last"])
    p.add_argument("--metric", choices=["r2", "mae"], default="r2")
    p.add_argument("--show-test", action="store_true", help="also save held-out test scores as a separate figure")
    p.add_argument("--x-fraction", action="store_true", help="x axis = layer / 24 instead of layer index")
    p.add_argument("--ymin", type=float, default=0.6, help="lower y limit for R2 plots")
    p.add_argument("--ymax", type=float, default=1.2, help="upper y limit for R2 plots (ticks every 0.2)")
    p.add_argument("--mark-layer", type=int, default=None, help="draw a dashed vertical line at this layer")
    p.add_argument("--mark-span", type=int, default=1,
                   help="shade mark-layer +- this many layers in light gray (0 = no shading)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_dir = args.results_dir / args.run_name
    out_dir = run_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    m = args.metric

    for pooling in args.poolings:
        # Load CSVs that have paper-style CV columns
        data = {}
        for v in args.variables:
            path = run_dir / f"probes_{v}_{pooling}.csv"
            if not path.exists():
                print(f"skip {v}/{pooling}: {path.name} not found")
                continue
            df = pd.read_csv(path)
            if f"cv_{m}_mean" not in df.columns:
                print(f"skip {v}/{pooling}: no cv_{m}_mean column (run with --paper-cv)")
                continue
            data[v] = df.sort_values("layer")
        if not data:
            continue

        # Main CV figure, plus optional test figure
        make_figure(data, m, "cv", args, out_dir / f"layers_{m}_{pooling}.png")
        if args.show_test:
            make_figure(data, m, "test", args, out_dir / f"layers_{m}_{pooling}_test.png")