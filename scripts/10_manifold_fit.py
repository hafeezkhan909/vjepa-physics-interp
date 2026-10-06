import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.manifold import fit_manifold
from src.metrics import deg_to_sincos, direction_metrics, scalar_metrics
from src.probe import load_split_data

UNIT = {"direction": "deg", "speed": "m/s", "acceleration": "m/s$^2$"}


def cmap_for(v):
    # Cyclic colormap for direction, sequential for scalars
    return "twilight" if v == "direction" else "viridis"


def spline_grid(m):
    # Dense points along the spline, in k-dim PCA coords
    u = np.linspace(0, 360, 721) if m.periodic else np.linspace(m.u[0], m.u[-1], 400)
    return m.s(u)


def chord_endpoints(m):
    # Example chord: 0° -> 180° for direction (across the loop), min -> max for scalars
    if m.periodic:
        i0 = int(np.argmin(np.abs(m.u - 0.0)))
        i1 = int(np.argmin(np.abs(m.u - 180.0)))
        return m.u[i0], m.u[i1]
    return m.u[0], m.u[-1]


def centroid_axes(m, n=3):
    # Top-n PCA axes of the centroids and their variance fractions
    Cc = m.C - m.C.mean(0)
    _, sv, Vt = np.linalg.svd(Cc, full_matrices=False)
    return m.C.mean(0), Vt[:n], (sv ** 2 / (sv ** 2).sum())[:n]


def plot_pca3d(m, Zte, yte, out, view="data"):
    # view='data': top-3 PCs of train activations; view='centroid': top-3 PCs of the centroids
    if view == "centroid":
        mu, A, frac = centroid_axes(m)

        def proj(P):
            return (P - mu) @ A.T

        lab, info = "cPC", f"centroid-PCA axes (cPC1-3 = {frac.sum() * 100:.1f}% of centroid var)"
    else:
        def proj(P):
            return P[:, :3]

        lab, info = "PC", f"data-PCA axes (PC1-3 = {m.evr[:3].sum() * 100:.1f}% of data var)"

    # Spline, example chord, test points and centroids in the chosen axes
    dense = proj(spline_grid(m))
    u0, u1 = chord_endpoints(m)
    a, b = m.s(np.array([u0, u1]))
    chord = proj(np.linspace(0, 1, 50)[:, None] * (b - a) + a)
    Zp, Cp = proj(Zte), proj(m.C)
    cmap = cmap_for(m.variable)

    fig = plt.figure(figsize=(14, 5.4))

    # Left: 3D view
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    sc = ax.scatter(*Zp.T, c=yte, cmap=cmap, s=5, alpha=0.35)
    ax.scatter(*Cp.T, c=m.u, cmap=cmap, s=28, edgecolor="k", lw=0.5)
    ax.plot(*dense.T, color="k", lw=1.8, label="spline")
    ax.plot(*chord.T, color="#d62728", lw=1.8, ls="--", label=f"chord {u0:.3g}→{u1:.3g}")
    ax.set_xlabel(f"{lab}1")
    ax.set_ylabel(f"{lab}2")
    ax.set_zlabel(f"{lab}3", labelpad=-2)
    ax.legend(fontsize=8, loc="upper left")

    # Right: first two axes, plus held-out bins if any
    ax2 = fig.add_subplot(1, 2, 2)
    fig.subplots_adjust(wspace=0.35)
    ax2.scatter(*Zp[:, :2].T, c=yte, cmap=cmap, s=5, alpha=0.35)
    ax2.scatter(*Cp[:, :2].T, c=m.u, cmap=cmap, s=28, edgecolor="k", lw=0.5)
    ax2.plot(*dense[:, :2].T, color="k", lw=1.8)
    ax2.plot(*chord[:, :2].T, color="#d62728", lw=1.8, ls="--")
    if len(m.held_out_u):
        h = proj(m.s(m.held_out_u))
        ax2.scatter(*h[:, :2].T, marker="x", color="k", s=30, label="held-out bins (spline)")
        ax2.legend(fontsize=8)
    ax2.set_xlabel(f"{lab}1")
    ax2.set_ylabel(f"{lab}2")
    ax2.grid(alpha=0.2)

    fig.colorbar(sc, ax=[ax, ax2], shrink=0.8, label=f"{m.variable} ({UNIT[m.variable]})")
    fig.suptitle(f"{m.variable} | L{m.layer} | {m.n_bins} bins | test activations, {info}")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def subspace_r2(m, Xtr, ytr, Xte, yte, variable):
    # Held-out ridge R²: full activations vs k-dim PCA projection
    is_dir = variable == "direction"
    target = deg_to_sincos(ytr) if is_dir else ytr
    metrics = direction_metrics if is_dir else scalar_metrics
    out = {}
    for name, A, B in [("full", Xtr, Xte), ("pca", m.to_pca(Xtr), m.to_pca(Xte))]:
        r = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(A, target)
        out[name] = float(metrics(yte, r.predict(B))["r2"])
    return out


def plot_evr(v, nb, layers, curves, out):
    # Cumulative variance: centroid curve (solid) vs all train activations (dashed), per layer
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    cmap = plt.get_cmap("viridis")
    for i, L in enumerate(layers):
        m, cev = curves[(v, nb, L)]
        c = cmap(i / max(1, len(layers) - 1))
        n = min(20, len(cev), len(m.evr))
        ax.plot(np.arange(1, n + 1), cev[:n], "-o", ms=3, color=c, label=f"L{L} centroid curve")
        ax.plot(np.arange(1, n + 1), np.cumsum(m.evr)[:n], "--", color=c, lw=1, alpha=0.6)
    ax.axhline(0.9, color="gray", ls=":", lw=1)
    ax.set_xlabel("number of PCs")
    ax.set_ylabel("cumulative variance explained")
    ax.set_title(f"{v} | {nb} bins | solid: centroid curve, dashed: all train activations", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_layers(v, nb, layers, curves, out):
    # Curve in centroid-PCA axes (cPC1-cPC2), one panel per layer
    fig, axs = plt.subplots(1, len(layers), figsize=(3 * len(layers), 3.4), squeeze=False)
    for a, L in zip(axs[0], layers):
        m, _ = curves[(v, nb, L)]
        mu, A, frac = centroid_axes(m, 2)
        dense = (spline_grid(m) - mu) @ A.T
        a.plot(*dense.T, color="k", lw=1)
        a.scatter(*((m.C - mu) @ A.T).T, c=m.u, cmap=cmap_for(v), s=14, edgecolor="k", lw=0.3)
        a.set_title(f"L{L} (cPC1-2: {frac.sum() * 100:.0f}%)", fontsize=9)
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"{v} | {nb} bins | curve in centroid-PCA axes (cPC1-cPC2) per layer", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variables", nargs="+", default=["direction", "speed", "acceleration"])
    p.add_argument("--layers", nargs="+", type=int, default=[8, 9, 15, 20, 22])
    p.add_argument("--pooling", default="mean")
    p.add_argument("--pca-dim", type=int, default=64)
    p.add_argument("--n-bins", nargs="+", type=int, default=[64, 32])
    p.add_argument("--drop-every", type=int, default=0, help="hold out every k-th bin from the spline fit (0 = none)")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--run-name", default="paper")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run = args.results_dir / args.run_name
    mdir, fdir = run / "manifold", run / "figs" / "manifold"
    mdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    tag = f"_drop{args.drop_every}" if args.drop_every > 1 else ""
    rows, curves = [], {}

    for v in args.variables:
        Xtr_all, ytr, Xte_all, yte = load_split_data(v, args.pooling)
        labels = np.sort(np.unique(np.concatenate([ytr, yte])))

        for nb in args.n_bins:
            for L in args.layers:
                Xtr_L = Xtr_all[:, L].astype(np.float64)
                Xte_L = Xte_all[:, L].astype(np.float64)

                # Fit PCA + centroids + spline on train, and save
                m = fit_manifold(v, L, Xtr_L, ytr, labels, args.pca_dim, nb, args.drop_every)
                stem = f"{v}_{args.pooling}_L{L:02d}_b{nb}{tag}"
                m.save(mdir / f"{stem}.npz")

                # Diagnostics: subspace R², centroid-curve dimensionality
                sub = subspace_r2(m, Xtr_L, ytr, Xte_L, yte, v)
                cev = np.linalg.svd(m.C - m.C.mean(0), compute_uv=False) ** 2
                cev = np.cumsum(cev) / cev.sum()
                rows.append(dict(variable=v, layer=L, n_bins=nb, n_centroids=len(m.u), held_out=len(m.held_out_u),
                                 curve_pc90=int(np.searchsorted(cev, 0.9) + 1),
                                 curve_var_pc1_3=float(cev[2]), data_var_pc1_3=float(m.evr[:3].sum()),
                                 min_count=int(m.counts.min()), r2_full=sub["full"], r2_pca=sub["pca"]))
                curves[(v, nb, L)] = (m, cev)
                print(f"{v:<13} L{L:02d} bins={nb:<3} centroids={len(m.u):<3} "
                      f"curve 90% var in {rows[-1]['curve_pc90']} PCs | min samples/bin={m.counts.min()} | "
                      f"held-out R² full={sub['full']:.3f} pca{args.pca_dim}={sub['pca']:.3f}")

                # Geometry plots in data-PCA and centroid-PCA axes
                Zte = m.to_pca(Xte_L)
                plot_pca3d(m, Zte, yte, fdir / f"pca3d_{stem}.png", view="data")
                plot_pca3d(m, Zte, yte, fdir / f"cpca3d_{stem}.png", view="centroid")

            # Per-variable, per-bins summaries across layers
            plot_evr(v, nb, args.layers, curves, fdir / f"evr_{v}_{args.pooling}_b{nb}{tag}.png")
            plot_layers(v, nb, args.layers, curves, fdir / f"layers_{v}_{args.pooling}_b{nb}{tag}.png")

    out = run / f"manifold_geometry{tag}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsaved {out} and figures in {fdir}")