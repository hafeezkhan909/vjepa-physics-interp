import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def nullspace_dims(folder: Path, variable: str, pooling: str, layer: int):
    # Dims removed until chance (paper hp run), and whether it hit max_iters
    stem = f"{variable}_{pooling}_L{layer:02d}_paper"
    npz, csv = folder / f"{stem}_basis.npz", folder / f"{stem}.csv"
    if not npz.exists():
        return None, False
    capped = not bool(pd.read_csv(csv).iloc[-1]["at_chance"])
    return int(np.load(npz)["V"].shape[1]), capped


def fmt_dims(d, capped):
    # '-' = no run, '*' = hit max_iters (lower bound)
    if d is None or pd.isna(d):
        return "  -"
    return f"{int(d):3d}{'*' if capped else ' '}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--run-name", default="paper")
    p.add_argument("--variables", nargs="*", default=["direction", "speed", "acceleration"])
    p.add_argument("--pooling", default="mean")
    p.add_argument("--nullspace-dir", default="nullspace", help="nullspace folder under results/<run-name>/")
    p.add_argument("--top", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run = args.results_dir / args.run_name
    rows = []
    for v in args.variables:
        # Join Item 1 probe scores with nullspace dims per layer
        probes = pd.read_csv(run / f"probes_{v}_{args.pooling}.csv").set_index("layer")
        recs = []
        for L in probes.index:
            dims, capped = nullspace_dims(run / args.nullspace_dir, v, args.pooling, L)
            p = probes.loc[L]
            recs.append(dict(variable=v, layer=int(L), cv_r2=p["cv_r2_mean"], cv_r2_std=p["cv_r2_std"],
                             test_r2=p["test_r2"], dims=dims, capped=capped))
        df = pd.DataFrame(recs)
        rows.append(df)

        # Per-layer table
        print(f"\n=== {v} ({args.pooling} pooling, nullspace hp = paper) ===")
        print(f"{'layer':>5} | {'CV R² (mean±std)':>17} | {'test R²':>7} | {'dims':>5}")
        for r in df.itertuples():
            print(f"{r.layer:>5} | {r.cv_r2:>8.3f} ± {r.cv_r2_std:.3f} | {r.test_r2:>7.3f} | "
                  f"{fmt_dims(r.dims, r.capped):>5}")

        # Range of nullspace dims across layers
        s = df.dropna(subset=["dims"])
        if len(s):
            lo, hi = s.loc[s["dims"].idxmin()], s.loc[s["dims"].idxmax()]
            print(f"  dims range: {int(lo['dims'])}-{int(hi['dims'])}  "
                  f"(min at L{int(lo.layer)}, max at L{int(hi.layer)})")

        # Top layers by probe score
        for col, name in [("cv_r2", "CV R²"), ("test_r2", "test R²")]:
            top = df.nlargest(args.top, col)
            print(f"  top-{args.top} by {name:<7}: " + ", ".join(f"L{int(r.layer)} ({r[col]:.4f})" for _, r in top.iterrows()))

    # Save all variables to one CSV
    out = run / "nullspace_summary.csv"
    pd.concat(rows).to_csv(out, index=False)
    print(f"\n'*' = hit max_iters (lower bound).  saved {out}")