import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.manifold import PERIOD, Manifold
from src.probe import load_split_data

COL = {"spline": "k", "chord": "#d62728", "multiprobe": "#1f77b4", "linear_interp": "#ff7f0e"}
UNIT = {"direction": "deg", "speed": "m/s", "acceleration": "m/s$^2$"}
METHODS = ["spline", "chord", "multiprobe"]
HELDOUT_METHODS = ["spline", "linear_interp", "multiprobe"]


def centroid_axes(m, n=2):
    # Top-n PCA axes of the centroids and their variance fractions
    Cc = m.C - m.C.mean(0)
    _, sv, Vt = np.linalg.svd(Cc, full_matrices=False)
    return m.C.mean(0), Vt[:n], (sv ** 2 / (sv ** 2).sum())[:n]


def layer_of(f):
    # Layer index from a filename like var_pool_Lxx_...
    return int(f.stem.split("_")[2][1:])


def plot_path(df, v, L, pair, out):
    # Three panels vs path fraction t: read vs intended, error, confidence / off-manifold distance
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    for k in METHODS:
        d = df[df["method"] == k].sort_values("t")
        t = d["t"].values
        if v == "direction":
            # Unwrap angles so the curves stay continuous across 0/360
            intended = np.degrees(np.unwrap(np.radians(d["intended"].values)))
            dev = (d["read_mean"].values - d["intended"].values + 180) % 360 - 180
            read = intended + dev
        else:
            intended, read = d["intended"].values, d["read_mean"].values
        axs[0].plot(t, read, "-o", ms=3, color=COL[k], label=k)
        axs[1].plot(t, d["err_mean"], "-o", ms=3, color=COL[k], label=k)
        third = d["confidence"] if v == "direction" else d["off_manifold"]
        axs[2].plot(t, third, "-o", ms=3, color=COL[k], label=k)
    axs[0].plot(t, intended, ":", color="gray", lw=2, label="intended")

    axs[0].set_ylabel(f"read value ({UNIT[v]})")
    axs[0].set_title("read vs intended along the path")
    axs[1].set_ylabel(f"error to intended ({UNIT[v]})")
    axs[1].set_title("error per step")
    if v == "direction":
        axs[2].set_ylabel("readout confidence ‖(sin, cos)‖")
        axs[2].axhline(1, color="gray", ls=":", lw=1)
        axs[2].set_title("confidence (≈0 = angle undefined)")
    else:
        axs[2].set_ylabel("off-manifold distance (PCA space)")
        axs[2].set_title("distance from the manifold")
    for a in axs:
        a.set_xlabel("path fraction t")
        a.grid(alpha=0.25)
        a.legend(fontsize=8)

    fig.suptitle(f"{v} | L{L} | path {pair} | eval: held-out evaluation probe")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_path_pca(m, Zte, yte, paths, v, L, pair, out):
    # Mean steered path of each method over the spline and test activations (centroid-PCA axes)
    mu, A, frac = centroid_axes(m, 2)

    def proj(Z):
        return (Z - mu) @ A.T

    g = np.linspace(0, PERIOD, 721) if m.periodic else np.linspace(m.u[0], m.u[-1], 400)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(*proj(Zte).T, c=yte, cmap="twilight" if m.periodic else "viridis", s=5, alpha=0.25)
    ax.plot(*proj(m.s(g)).T, color="gray", lw=4, alpha=0.35, label="manifold (spline)")
    for k in METHODS:
        key = f"{pair}|{k}"
        if key in paths:
            ax.plot(*proj(paths[key]).T, "-o", ms=3, color=COL[k], lw=1.6, label=f"{k} (mean steered path)")

    ax.set_xlabel("cPC1")
    ax.set_ylabel("cPC2")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    ax.set_title(f"{v} | L{L} | path {pair} | centroid-PCA axes ({frac.sum() * 100:.0f}% centroid var)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_summary(S, v, out):
    # Bars: mean path error per method x layer, one panel per path
    pairs = list(S["pair"].unique())
    fig, axs = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 4), squeeze=False)
    for ax, pair in zip(axs[0], pairs):
        d = S[S["pair"] == pair]
        layers = sorted(d["layer"].unique())
        x, w = np.arange(len(layers)), 0.27
        for i, k in enumerate(METHODS):
            vals = [d[(d["layer"] == L) & (d["method"] == k)]["path_err"].values[0] for L in layers]
            ax.bar(x + (i - 1) * w, vals, w, color=COL[k], label=k)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{L}" for L in layers])
        ax.set_ylabel(f"mean path error ({UNIT[v]})")
        ax.set_title(f"{v} | path {pair}")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)  # headroom so the legend never covers a bar
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=8, ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_heldout(H, variables, out):
    # Bars: error at held-out bins per method x layer, one panel per variable
    vs = [v for v in variables if v in set(H["variable"])]
    fig, axs = plt.subplots(1, len(vs), figsize=(5.5 * len(vs), 4), squeeze=False)
    for ax, v in zip(axs[0], vs):
        d = H[H["variable"] == v]
        layers = sorted(d["layer"].unique())
        x, w = np.arange(len(layers)), 0.27
        for i, k in enumerate(HELDOUT_METHODS):
            vals = [d[(d["layer"] == L) & (d["method"] == k)]["err"].values[0] for L in layers]
            ax.bar(x + (i - 1) * w, vals, w, color=COL[k], label=k)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{L}" for L in layers])
        ax.set_ylabel(f"error at held-out bins ({UNIT[v]})")
        ax.set_title(f"{v} | held-out bins")
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variables", nargs="+", default=["direction", "speed", "acceleration"])
    p.add_argument("--layers", nargs="*", type=int, default=None, help="default: all found")
    p.add_argument("--n-bins", type=int, default=32)
    p.add_argument("--edit", default="replace")
    p.add_argument("--drop-every", type=int, default=4, help="which _drop runs to use for heldout.png")
    p.add_argument("--pooling", default="mean")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--run-name", default="paper")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run = args.results_dir / args.run_name
    sdir, fdir = run / "manifold_steer", run / "figs" / "manifold_steer"
    fdir.mkdir(parents=True, exist_ok=True)
    summary = []

    # Path steering plots per variable / layer / path
    for v in args.variables:
        files = sorted(sdir.glob(f"{v}_{args.pooling}_L*_b{args.n_bins}_{args.edit}.csv"))
        if not files:
            print(f"skip {v}: no CSVs")
            continue
        _, _, Xte_all, yte = load_split_data(v, args.pooling)

        found = False
        for f in files:
            L = layer_of(f)
            if args.layers and L not in args.layers:
                continue
            found = True
            df = pd.read_csv(f)
            df = df[df["pair"] != "heldout"]

            # Manifold, test activations in PCA coords, and mean steered paths
            m = Manifold.load(run / "manifold" / f"{v}_{args.pooling}_L{L:02d}_b{args.n_bins}.npz")
            Zte = m.to_pca(Xte_all[:, L].astype(np.float64))
            pz = sdir / f"{f.stem}_paths.npz"
            paths = dict(np.load(pz)) if pz.exists() else {}

            for pair, d in df.groupby("pair"):
                tag = pair.replace(":", "to")
                plot_path(d, v, L, pair, fdir / f"path_{v}_L{L:02d}_{tag}.png")
                plot_path_pca(m, Zte, yte, paths, v, L, pair, fdir / f"pathpca_{v}_L{L:02d}_{tag}.png")

                # Summary row per method
                for k in METHODS:
                    dk = d[d["method"] == k]
                    summary.append(dict(variable=v, layer=L, pair=pair, method=k,
                                        path_err=dk["err_mean"].mean(), end_err=dk["err_mean"].iloc[-1],
                                        max_err=dk["err_mean"].max(), off_manifold=dk["off_manifold"].mean(),
                                        mid_conf=dk.sort_values("t")["confidence"].iloc[len(dk) // 2]))
        if found:
            S = pd.DataFrame([s for s in summary if s["variable"] == v])
            plot_summary(S, v, fdir / f"summary_{v}.png")

    # Held-out bins, from the _drop runs
    hrows = []
    for v in args.variables:
        for f in sorted(sdir.glob(f"{v}_{args.pooling}_L*_b{args.n_bins}_drop{args.drop_every}_replace.csv")):
            d = pd.read_csv(f)
            d = d[d["pair"] == "heldout"]
            for k, g in d.groupby("method"):
                hrows.append(dict(variable=v, layer=layer_of(f), method=k, err=g["err_mean"].mean()))
    if hrows:
        H = pd.DataFrame(hrows)
        plot_heldout(H, args.variables, fdir / "heldout.png")
        print("\nHELD-OUT BINS (mean error to the held-out value)")
        print(H.pivot_table(index=["variable", "layer"], columns="method", values="err").round(3).to_string())

    # Save and print the path-steering summary
    if summary:
        S = pd.DataFrame(summary)
        S.to_csv(run / "manifold_steer_summary.csv", index=False)
        print("\nPATH STEERING (mean error along the path)")
        print(S.pivot_table(index=["variable", "pair", "layer"], columns="method", values="path_err").round(3).to_string())
    print(f"\nfigures in {fdir}")