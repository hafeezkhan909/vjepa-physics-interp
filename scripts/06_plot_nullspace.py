import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
YLABEL = {"r2": "test R$^2$", "acc15": "accuracy within 15° (%)"}
UNITS = {"speed": "m/s", "direction": "deg", "acceleration": "m/s$^2$"}
STYLE = {"paper": "-", "item1": "--"}  # line style per hparam source
VCOL = {"direction": "#d62728", "speed": "#1f77b4", "acceleration": "#2ca02c"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="results/paper/nullspace", help="folder with the nullspace CSVs")
    p.add_argument("--out-dir", default="results/paper/figs", help="folder for the figures")
    p.add_argument("--variables", nargs="*", default=["direction", "speed", "acceleration"])
    p.add_argument("--pooling", default="mean")
    p.add_argument("--metric", choices=["r2", "mae", "acc15"], default="r2",
                   help="acc15 = %% within 15° (direction only)")
    p.add_argument("--x", choices=["dims", "probe"], default="dims",
                   help="x axis: dims removed, or orthogonal probe number (as in the paper)")
    p.add_argument("--layers", nargs="*", type=int, default=None, help="only these layers (default: all found)")
    p.add_argument("--hp", nargs="*", default=["paper"])
    p.add_argument("--layer-summary", action="store_true",
                   help="also plot number of orthogonal probes until chance vs layer (one line per variable)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Relative paths are resolved from the repo root
    in_dir = Path(args.in_dir) if Path(args.in_dir).is_absolute() else ROOT / args.in_dir
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    m = args.metric

    # Metric vs dims removed: one figure per variable
    for v in args.variables:
        # Collect matching runs; filename = var_pool_Lxx_hp[_stop-xx][_rm-xx]
        curves = []
        for f in sorted(in_dir.glob(f"{v}_{args.pooling}_L*_*.csv")):
            parts = f.stem.split("_")
            layer, hp, tags = int(parts[2][1:]), parts[3], parts[4:]
            if hp not in args.hp or (args.layers and layer not in args.layers):
                continue
            df = pd.read_csv(f)
            if f"test_{m}" not in df.columns or df[f"test_{m}"].isna().all():
                print(f"skip {f.name}: no test_{m}")
                continue
            curves.append((layer, " ".join([hp, *tags]), df))
        if not curves:
            print(f"skip {v}: no matching CSVs in {in_dir}")
            continue

        # Many curves -> color by layer with a colorbar; few -> labeled lines
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        layers = sorted({c[0] for c in curves})
        many = len(curves) > 8
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(min(layers), max(layers)) if len(layers) > 1 else None
        for layer, label, df in curves:
            xs = df["iter"] if args.x == "probe" else df["dims_removed"]
            kw = dict(ls=STYLE.get(label.split()[0], "-"), lw=1.3, marker="o", ms=2)
            if many and norm is not None:
                ax.plot(xs, df[f"test_{m}"], color=cmap(norm(layer)), **kw)
            else:
                ax.plot(xs, df[f"test_{m}"], label=f"L{layer} {label}", **kw)
        if many and norm is not None:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            fig.colorbar(sm, ax=ax, label="layer")
        # ax.legend(fontsize=8)

        ax.set_title(v)
        ax.set_xlabel("orthogonal probe number" if args.x == "probe" else "dims removed")
        ax.set_ylabel(YLABEL[m] if m in YLABEL else f"test MAE ({UNITS[v]})")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        out = out_dir / f"nullspace_{v}_{m}_x-{args.x}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print("saved", out)

    # Probes until chance vs layer: color = variable, line style = hp
    if args.layer_summary:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for v in args.variables:
            for hp in args.hp:
                pts = []
                for f in sorted(in_dir.glob(f"{v}_{args.pooling}_L*_{hp}*.csv")):
                    parts = f.stem.split("_")
                    # Only the default stop rule / removal mode
                    if parts[3] != hp or any(t.startswith(("stop-", "rm-")) for t in parts[4:]):
                        continue
                    if args.layers and int(parts[2][1:]) not in args.layers:
                        continue
                    df = pd.read_csv(f)
                    # Probes removed before reaching chance (all of them if capped at max-iters)
                    capped = not bool(df.iloc[-1]["at_chance"])
                    probes = len(df) if capped else len(df) - 1
                    pts.append((int(parts[2][1:]), probes, capped))
                if not pts:
                    continue
                pts.sort()
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                ls = STYLE.get(hp, "-") if len(args.hp) > 1 else "-"
                ax.plot(xs, ys, color=VCOL.get(v), ls=ls, lw=2, marker="o", ms=4,
                        label=v if len(args.hp) == 1 else f"{v} ({hp})")
        ax.set_xlabel("layer")
        ax.set_ylabel("orthogonal probes until chance")
        ax.set_title("Probes until chance vs layer")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = out_dir / "nullspace_probes_vs_layer.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print("saved", out)