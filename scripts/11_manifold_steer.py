import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.manifold import PERIOD, Manifold, path_u
from src.probe import load_split_data
from src.steering import decode, encode, err, load_probes, steer, train_eval_probe

METHODS = ["spline", "chord", "multiprobe"]


def resolve(tok, m):
    # Parse a path endpoint: 'min' / 'max' = spline ends; scalars are clipped to the spline range
    if tok == "min":
        return float(m.u[0])
    if tok == "max":
        return float(m.u[-1])
    v = float(tok)
    if not m.periodic and not (m.u[0] <= v <= m.u[-1]):
        print(f"  [note] {v:g} outside the spline range [{m.u[0]:.3g}, {m.u[-1]:.3g}] -> clipped")
        v = float(np.clip(v, m.u[0], m.u[-1]))
    return v


def dense_curve(m, n=2000):
    # Dense points along the spline, in PCA coords
    g = np.linspace(0, PERIOD, n, endpoint=False) if m.periodic else np.linspace(m.u[0], m.u[-1], n)
    return m.s(g)


def off_manifold(Z, curve):
    # Mean distance (PCA space) from each row of Z to the nearest spline point
    d2 = (Z ** 2).sum(1)[:, None] - 2 * Z @ curve.T + (curve ** 2).sum(1)[None]
    return float(np.sqrt(np.maximum(d2.min(1), 0)).mean())


def edit_size(Xs, X):
    # Relative size of the edit
    return float(np.linalg.norm(Xs - X) / np.linalg.norm(X))


def readout(variable, P, intended):
    # Decode eval-probe outputs; error to the intended value, mean / spread of reads
    pred = decode(variable, P)
    e = err(variable, pred, intended)
    out = dict(err_mean=float(e.mean()), err_median=float(np.median(e)))
    if variable == "direction":
        # Circular mean / spread; confidence = ||(sin, cos)||, near 0 when the angle is undefined
        r = np.radians(pred)
        s, c = np.sin(r).mean(), np.cos(r).mean()
        out["read_mean"] = float(np.degrees(np.arctan2(s, c)) % PERIOD)
        out["read_spread"] = float(np.degrees(np.sqrt(-2 * np.log(max(np.hypot(s, c), 1e-12)))))
        out["confidence"] = float(np.linalg.norm(P, axis=1).mean())
    else:
        out["read_mean"], out["read_spread"], out["confidence"] = float(pred.mean()), float(pred.std()), np.nan
    return out


def neighbour_linear(m, u):
    # Linear interpolation between the two kept centroids around u (wraps for direction)
    us, C = m.u, m.C
    if m.periodic:
        us_ext = np.concatenate([us - PERIOD, us, us + PERIOD])
        C_ext = np.vstack([C, C, C])
        u = u % PERIOD
    else:
        us_ext, C_ext = us, C
    j = int(np.searchsorted(us_ext, u))
    a, b = us_ext[j - 1], us_ext[j]
    w = (u - a) / (b - a)
    return (1 - w) * C_ext[j - 1] + w * C_ext[j]


def run_path(pair, v, L, nb, m, X_all, yte, W, b, eval_probe, curve, width, args):
    # Steer along one start->target path with spline, chord and multi-probe
    We, be = eval_probe
    u0, u1 = (resolve(t, m) for t in pair.split(":"))
    us, ts = path_u(u0, u1, args.T, m.periodic), np.linspace(0, 1, args.T + 1)

    # Samples to steer: all test samples, or only those in the start bin
    sel = err(v, yte, u0) <= width if args.sample_set == "start" else np.ones(len(yte), bool)
    X = X_all[sel]
    Z0 = m.to_pca(X)
    sA, sB = m.s(u0), m.s(u1)

    rows, mean_paths = [], {k: [] for k in METHODS}
    for t, u in zip(ts, us):
        intended = u % PERIOD if m.periodic else u

        # Spline / chord: set PCA coords to the target point (replace) or move by target - s(u0) (shift)
        targets = {"spline": m.s(u), "chord": (1 - t) * sA + t * sB}
        Xs = {}
        for k, zt in targets.items():
            if args.edit == "replace":
                Xs[k] = X + (zt[None] - Z0) @ m.comps
            else:
                Xs[k] = X + (zt - sA)[None] @ m.comps

        # Multi-probe: Part 1 subspace steering to the intended value
        Xs["multiprobe"] = steer(X, W, b, encode(v, intended)[0])[0]

        # Read out each method with the eval probe
        for k in METHODS:
            Zs = m.to_pca(Xs[k])
            mean_paths[k].append(Zs.mean(0))
            rows.append(dict(variable=v, layer=L, n_bins=nb, edit=args.edit, pair=pair, method=k,
                             t=t, intended=intended, n=int(sel.sum()),
                             **readout(v, Xs[k] @ We.T + be, intended),
                             off_manifold=off_manifold(Zs, curve), edit_rel=edit_size(Xs[k], X)))

    print(f"  path {pair}  ({u0:.3g} → {u1:.3g}, T={args.T}, n={int(sel.sum())})")
    print_path_table(pd.DataFrame(rows), v)
    return rows, {k: np.array(p) for k, p in mean_paths.items()}


def print_path_table(df, v):
    # Path err, end err, max err, mid-path confidence, off-manifold distance, edit size per method
    unit = "°" if v == "direction" else ""
    print(f"    {'method':<11}{'path err':>10}{'end err':>10}{'max err':>10}"
          f"{'mid conf':>10}{'off-man':>10}{'edit':>8}")
    for k in METHODS:
        d = df[df["method"] == k]
        mid = d.iloc[len(d) // 2]
        conf = f"{mid['confidence']:>10.3f}" if v == "direction" else f"{'-':>10}"
        print(f"    {k:<11}{d['err_mean'].mean():>9.3f}{unit}{d['err_mean'].iloc[-1]:>9.3f}{unit}"
              f"{d['err_mean'].max():>9.3f}{unit}{conf}{d['off_manifold'].mean():>10.3f}"
              f"{d['edit_rel'].mean():>8.3f}")


def run_heldout(v, L, nb, m, X_all, W, b, eval_probe, curve):
    # Steer straight to each held-out bin value: spline, linear interp of neighbours, multi-probe
    We, be = eval_probe
    if not len(m.held_out_u):
        print("  [heldout] manifold has no held-out bins (fit with --drop-every k)")
        return []

    X, Z0 = X_all, m.to_pca(X_all)
    rows = []
    for u in m.held_out_u:
        intended = u % PERIOD if m.periodic else u
        cand = {"spline": X + (m.s(u)[None] - Z0) @ m.comps,
                "linear_interp": X + (neighbour_linear(m, u)[None] - Z0) @ m.comps,
                "multiprobe": steer(X, W, b, encode(v, intended)[0])[0]}
        for k, Xs in cand.items():
            rows.append(dict(variable=v, layer=L, n_bins=nb, edit="replace", pair="heldout",
                             method=k, t=1.0, intended=intended, n=len(X),
                             **readout(v, Xs @ We.T + be, intended),
                             off_manifold=off_manifold(m.to_pca(Xs), curve), edit_rel=edit_size(Xs, X)))

    # Mean error to the held-out values per method
    unit = "°" if v == "direction" else ""
    dh = pd.DataFrame(rows)
    print(f"  held-out bins ({len(m.held_out_u)} values not used to fit the spline):")
    for k in ["spline", "linear_interp", "multiprobe"]:
        print(f"    {k:<14} mean err to held-out value = {dh[dh['method'] == k]['err_mean'].mean():.3f}{unit}")
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variables", nargs="+", default=["direction", "speed", "acceleration"])
    p.add_argument("--layers", nargs="+", type=int, default=[8, 9, 15, 20, 22])
    p.add_argument("--n-bins", nargs="+", type=int, default=[32])
    p.add_argument("--drop-every", type=int, default=0, help="use the manifolds fit with this --drop-every")
    p.add_argument("--pairs", nargs="*", default=None,
                   help="start:target, e.g. 0:180 0:90 (default: direction 0:180 0:90; scalars min:max)")
    p.add_argument("--T", type=int, default=20, help="number of path steps")
    p.add_argument("--edit", choices=["replace", "shift"], default="replace")
    p.add_argument("--sample-set", choices=["all", "start"], default="all",
                   help="steer all test samples, or only those whose label lies in the start bin")
    p.add_argument("--mp-k", type=int, default=0, help="probes for multi-probe steering (0 = all)")
    p.add_argument("--heldout", action="store_true", help="also run the held-out-bin test (needs --drop-every)")
    p.add_argument("--pooling", default="mean")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--run-name", default="paper")
    p.add_argument("--probe-dir", default="steer_orth", help="Part 1 steering probes (W, b) for multi-probe")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run = args.results_dir / args.run_name
    out_dir = run / "manifold_steer"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_drop{args.drop_every}" if args.drop_every > 1 else ""
    line = "=" * 104

    for v in args.variables:
        _, _, Xte_all, yte = load_split_data(v, args.pooling)
        pairs = args.pairs or (["0:180", "0:90"] if v == "direction" else ["min:max"])

        for nb in args.n_bins:
            for L in args.layers:
                # Load the train-fit manifold and the test activations for this layer
                stem = f"{v}_{args.pooling}_L{L:02d}_b{nb}{tag}"
                m = Manifold.load(run / "manifold" / f"{stem}.npz")
                X_all = Xte_all[:, L].astype(np.float64)

                # Eval probe: ridge on unsteered test activations (same as Part 1)
                We, be, m_eval, info = train_eval_probe(X_all, yte, v, "ridge", 0, "cpu")

                # Part 1 steering probes, first K used for multi-probe
                W, b = load_probes(run / args.probe_dir, v, args.pooling, L)
                K = W.shape[0] if args.mp_k <= 0 else min(args.mp_k, W.shape[0])

                curve = dense_curve(m)
                width = (PERIOD if m.periodic else (m.u[-1] - m.u[0])) / nb  # one bin width
                print(f"\n{line}\nManifold steering | {v}  L{L}  {nb} bins{tag}  edit={args.edit}  "
                      f"samples={args.sample_set}\n  eval probe (test, {info}): R²={m_eval['r2']:.3f}  "
                      f"| multi-probe K={K}  | spline knots={len(m.u)}\n{line}")

                # Path steering for each start:target pair
                rows, paths = [], {}
                for pair in pairs:
                    r, paths[pair] = run_path(pair, v, L, nb, m, X_all, yte, W[:K], b[:K],
                                              (We, be), curve, width, args)
                    rows += r

                # Optional held-out-bin test
                if args.heldout:
                    rows += run_heldout(v, L, nb, m, X_all, W[:K], b[:K], (We, be), curve)

                # Save per-step rows and mean PCA paths
                csv = out_dir / f"{stem}_{args.edit}.csv"
                pd.DataFrame(rows).to_csv(csv, index=False)
                np.savez(out_dir / f"{stem}_{args.edit}_paths.npz",
                         **{f"{p}|{k}": a for p, d in paths.items() for k, a in d.items()})
                print(f"  saved {csv}")