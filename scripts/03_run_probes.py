import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import VARIABLES
from src.probe import ProbeConfig, probe_layers


def parse_args():
    d = ProbeConfig()  # defaults come from ProbeConfig
    p = argparse.ArgumentParser()
    p.add_argument("variables", nargs="*", default=VARIABLES)
    p.add_argument("--poolings", nargs="*", default=["mean", "last"])
    p.add_argument("--lrs", nargs="*", type=float, default=list(d.lrs))
    p.add_argument("--wds", nargs="*", type=float, default=list(d.wds))
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--patience", type=int, default=d.patience)
    p.add_argument("--val-frac", type=float, default=d.val_frac, help="inner val fraction when --cv-folds 0")
    p.add_argument("--cv-folds", type=int, default=d.cv_folds, help="0 = single val split; k = stratified k-fold CV in train")
    p.add_argument("--batch-size", type=int, default=d.batch_size, help="0 = full batch")
    p.add_argument("--standardize-features", action="store_true", help="z-score features (off = paper)")
    p.add_argument("--standardize-targets", action="store_true", help="z-score scalar targets (off = paper)")
    p.add_argument("--paper-cv", action="store_true", help="also compute paper-style CV (best config per fold, mean+-std)")
    p.add_argument("--paper-cv-folds", type=int, default=d.paper_cv_folds)
    p.add_argument("--first-layer", type=int, default=d.first_layer, help="1 = skip layer 0 (patch embedding)")
    p.add_argument("--resume", action="store_true", help="skip layers already saved in the run's CSVs (config must match)")
    p.add_argument("--save-probes", action="store_true", help="save the final probe (weights + normalization) per layer to <run>/checkpoints/")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--run-name", default="baseline", help="results go to <results-dir>/<run-name>/")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default=d.device)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = ProbeConfig(
        lrs=tuple(args.lrs),
        wds=tuple(args.wds),
        epochs=args.epochs,
        patience=args.patience,
        val_frac=args.val_frac,
        cv_folds=args.cv_folds,
        batch_size=args.batch_size,
        standardize_features=args.standardize_features,
        standardize_targets=args.standardize_targets,
        paper_cv=args.paper_cv,
        paper_cv_folds=args.paper_cv_folds,
        first_layer=args.first_layer,
        seed=args.seed,
        device=args.device,
    )
    print(cfg)

    # Report which device is actually used
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        print(f"[device] running on GPU: {torch.cuda.get_device_name(torch.device(cfg.device))} ({cfg.device})")
    else:
        print(f"[device] running on CPU (requested '{cfg.device}', cuda available: {torch.cuda.is_available()})")

    # Save config; on --resume, refuse to mix runs with different configs
    out_dir = args.results_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config.txt"
    if args.resume and cfg_path.exists() and cfg_path.read_text() != repr(cfg):
        sys.exit(f"--resume: config differs from {cfg_path}; use identical args or a new --run-name")
    cfg_path.write_text(repr(cfg))

    # Probe every layer for each variable / pooling
    for v in args.variables:
        for p in args.poolings:
            csv_path = out_dir / f"probes_{v}_{p}.csv"
            if csv_path.exists() and not args.resume:
                print(f"[note] {csv_path.name} exists and will be overwritten (use --resume to continue it)")
            ckpt_dir = out_dir / "checkpoints" if args.save_probes else None
            df = probe_layers(v, p, cfg, csv_path=csv_path, resume=args.resume, ckpt_dir=ckpt_dir,
                              cache_dir=args.cache_dir)

            # Best layer picked on inner-val, not test
            best = df.loc[df["val_mse"].idxmin()]
            print(f"==> {v}/{p}: best layer by val = {int(best['layer'])}, "
                  f"test_r2={best['test_r2']:.3f}, test_mae={best['test_mae']:.3f}\n")