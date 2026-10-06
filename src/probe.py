import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.metrics import deg_to_sincos, direction_metrics, scalar_metrics

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
N_LAYERS = 25  # patch embedding + 24 blocks


@dataclass
class ProbeConfig:
    lrs: tuple = (1e-4, 3e-4, 1e-3, 3e-3, 5e-3)  # paper's grid
    wds: tuple = (0.01, 0.1, 0.4, 0.8)           # paper's grid
    epochs: int = 1000         # max epochs (not specified in the paper)
    patience: int = 50         # early stopping on val MSE
    val_frac: float = 0.2      # inner-val fraction of the train split (used when cv_folds == 0)
    cv_folds: int = 0          # 0 = single inner val split; k > 0 = stratified k-fold CV in train
    batch_size: int = 128      # 0 = full batch
    standardize_features: bool = False  # z-score features with train stats (not in the paper)
    standardize_targets: bool = False   # z-score scalar targets (not in the paper; direction untouched)
    first_layer: int = 1       # layer 0 (patch embedding) is skipped
    paper_cv: bool = False     # also compute paper-style CV: best config per fold, mean +- std
    paper_cv_folds: int = 5
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_split_data(variable: str, pooling: str, cache_dir: Path = CACHE):
    # Cached features [N, 25, 1024] and labels, split into train / test by clip id
    d = np.load(Path(cache_dir) / f"feats_{variable}.npz")
    X = d[f"feats_{pooling}"].astype(np.float32)
    y = d["target"]
    row = {int(i): k for k, i in enumerate(d["ids"])}
    with open(Path(cache_dir) / f"splits_{variable}.json") as f:
        s = json.load(f)
    tr = np.array([row[i] for i in s["train"]])
    te = np.array([row[i] for i in s["test"]])
    return X[tr], y[tr], X[te], y[te]


def _stats(a: np.ndarray, eps: float = 1e-6):
    # Column mean / std for z-scoring (constant columns get std 1)
    mean = a.mean(axis=0, keepdims=True)
    std = a.std(axis=0, keepdims=True)
    std[std < eps] = 1.0
    return mean, std


def _feat_stats(a: np.ndarray, cfg):
    # Feature stats if standardizing, else identity
    return _stats(a) if cfg.standardize_features else (0.0, 1.0)


def _target_stats(a: np.ndarray, zt: bool):
    # Target stats if standardizing, else identity
    return _stats(a) if zt else (0.0, 1.0)


def _t(a, cfg):
    return torch.tensor(np.asarray(a), dtype=torch.float32, device=cfg.device)


def _f32(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32)


def _fold_tensors(X, T, tr_i, va_i, cfg, zt):
    # Normalized train / val tensors, using stats from the train part only
    xm, xs = _feat_stats(X[tr_i], cfg)
    tm, ts = _target_stats(T[tr_i], zt)
    a, ta = _t((X[tr_i] - xm) / xs, cfg), _t((T[tr_i] - tm) / ts, cfg)
    b, tb = _t((X[va_i] - xm) / xs, cfg), _t((T[va_i] - tm) / ts, cfg)
    return a, ta, b, tb, tm, ts


def _metrics(y, pred, is_dir):
    # Circular metrics on (sin, cos) for direction, scalar metrics otherwise
    return direction_metrics(y, pred) if is_dir else scalar_metrics(y, pred[:, 0])


def _stratified_kfold(n_folds, strata, seed):
    return list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(np.zeros(len(strata)), strata))


def _selection_splits(strata, cfg):
    # One stratified inner train/val split, or k stratified folds, inside train
    if cfg.cv_folds > 0:
        return _stratified_kfold(cfg.cv_folds, strata, cfg.seed)
    idx = np.arange(len(strata))
    return [train_test_split(idx, test_size=cfg.val_frac, stratify=strata, random_state=cfg.seed)]


def _train(Xtr, Ttr, Xva, Tva, lr, wd, cfg, epochs):
    # AdamW + MSE on nn.Linear; with a val set: early stopping, returns best (val_mse, epoch, val preds);
    # without (Xva=None): trains exactly `epochs` epochs
    torch.manual_seed(cfg.seed)
    model = torch.nn.Linear(Xtr.shape[1], Ttr.shape[1]).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = len(Xtr)
    bs = cfg.batch_size or n
    best, best_ep, bad, best_pred = float("inf"), epochs, 0, None

    for ep in range(1, epochs + 1):
        # One epoch of minibatch updates
        perm = torch.randperm(n, device=cfg.device)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            loss = F.mse_loss(model(Xtr[idx]), Ttr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # Early stopping on val MSE
        if Xva is not None:
            with torch.no_grad():
                out = model(Xva)
                vl = F.mse_loss(out, Tva).item()
            if vl < best - 1e-8:
                best, best_ep, bad, best_pred = vl, ep, 0, out.cpu().numpy()
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    return model, best, best_ep, best_pred


def _select_config(data, cfg):
    # Grid search: lowest mean val MSE over splits; epochs = mean best epoch
    best = None
    for lr, wd in product(cfg.lrs, cfg.wds):
        res = [_train(a, ta, b, tb, lr, wd, cfg, cfg.epochs)[1:3] for a, ta, b, tb in data]
        vmse = float(np.mean([r[0] for r in res]))
        ep = max(1, int(round(np.mean([r[1] for r in res]))))
        if best is None or vmse < best[0]:
            best = (vmse, lr, wd, ep)
    return best


def _paper_cv(X, Ttr, ytr, folds, cfg, is_dir, zt) -> dict:
    # Paper-style CV inside train: per fold, best config's score on that fold; mean +- std over folds
    scores = []
    for tr_i, va_i in folds:
        a, ta, b, tb, tm, ts = _fold_tensors(X, Ttr, tr_i, va_i, cfg, zt)

        # Best config on this fold (by val MSE at its best epoch)
        best = None
        for lr, wd in product(cfg.lrs, cfg.wds):
            _, vmse, ep, pred = _train(a, ta, b, tb, lr, wd, cfg, cfg.epochs)
            if best is None or vmse < best[0]:
                best = (vmse, lr, wd, ep, pred)
        _, lr, wd, ep, pred = best

        m = _metrics(ytr[va_i], pred * ts + tm, is_dir)
        scores.append((m["r2"], m["mae"], lr, wd, ep))

    r2, mae = np.array([s[0] for s in scores]), np.array([s[1] for s in scores])
    return {
        "cv_r2_mean": float(r2.mean()),
        "cv_r2_std": float(r2.std()),
        "cv_mae_mean": float(mae.mean()),
        "cv_mae_std": float(mae.std()),
        "cv_lrs": ",".join(f"{s[2]:g}" for s in scores),
        "cv_wds": ",".join(f"{s[3]:g}" for s in scores),
        "cv_epochs": ",".join(str(s[4]) for s in scores),
    }


def _ckpt_path(ckpt_dir, variable: str, pooling: str, layer: int) -> Path:
    return Path(ckpt_dir) / f"{variable}_{pooling}_L{layer:02d}.pt"


def _save_ckpt(ckpt_dir, variable, pooling, layer, model, xm, xs, tm, ts, is_dir, lr, wd, ep, vmse, cfg):
    # Final probe (all of train, chosen config) + its normalization
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "variable": variable, "pooling": pooling, "layer": layer,
            "weight": model.weight.detach().cpu(),  # [out, 1024]
            "bias": model.bias.detach().cpu(),      # [out]
            "x_mean": _f32(xm), "x_std": _f32(xs),  # input: (x - x_mean) / x_std
            "t_mean": _f32(tm), "t_std": _f32(ts),  # output: pred * t_std + t_mean
            "output": "sincos" if is_dir else "scalar",
            "lr": lr, "wd": wd, "epochs": ep, "val_mse": vmse, "cfg": repr(cfg),
        },
        _ckpt_path(ckpt_dir, variable, pooling, layer),
    )


def _load_resume(csv_path, ckpt_dir, variable, pooling):
    # Rows and layers already done; with checkpoints, a layer counts only if its checkpoint exists
    prev = pd.read_csv(csv_path)
    rows = prev.to_dict("records")
    done = set(prev["layer"].astype(int))
    if ckpt_dir is not None:
        done = {l for l in done if _ckpt_path(ckpt_dir, variable, pooling, l).exists()}
        rows = [r for r in rows if int(r["layer"]) in done]
    print(f"[resume] {Path(csv_path).name}: layers already done: {sorted(done)}")
    return rows, done


def probe_layers(
    variable: str, pooling: str, cfg: ProbeConfig, csv_path=None, resume: bool = False, ckpt_dir=None,
    cache_dir: Path = CACHE,
) -> pd.DataFrame:
    Xtr, ytr, Xte, yte = load_split_data(variable, pooling, cache_dir)

    # Targets: (sin, cos) for direction, scalar column otherwise
    is_dir = variable == "direction"
    Ttr = deg_to_sincos(ytr) if is_dir else ytr[:, None].astype(np.float32)
    zt = cfg.standardize_targets and not is_dir

    # Splits inside train, stratified on the label
    strata = pd.factorize(ytr)[0]
    splits = _selection_splits(strata, cfg)
    cv_splits = _stratified_kfold(cfg.paper_cv_folds, strata, cfg.seed) if cfg.paper_cv else None

    rows, done = [], set()
    if resume and csv_path is not None and Path(csv_path).exists():
        rows, done = _load_resume(csv_path, ckpt_dir, variable, pooling)

    for layer in range(cfg.first_layer, N_LAYERS):
        if layer in done:
            continue
        X, Xt = Xtr[:, layer], Xte[:, layer]

        # Select (lr, wd, epochs) on the inner splits
        data = [_fold_tensors(X, Ttr, tr_i, va_i, cfg, zt)[:4] for tr_i, va_i in splits]
        vmse, lr, wd, ep = _select_config(data, cfg)

        # Refit on full train with the chosen config, evaluate once on test
        xm, xs = _feat_stats(X, cfg)
        tm, ts = _target_stats(Ttr, zt)
        model, _, _, _ = _train(_t((X - xm) / xs, cfg), _t((Ttr - tm) / ts, cfg), None, None, lr, wd, cfg, ep)
        with torch.no_grad():
            pred = model(_t((Xt - xm) / xs, cfg)).cpu().numpy() * ts + tm
        m = _metrics(yte, pred, is_dir)

        if ckpt_dir is not None:
            _save_ckpt(ckpt_dir, variable, pooling, layer, model, xm, xs, tm, ts, is_dir, lr, wd, ep, vmse, cfg)

        rows.append(
            {
                "variable": variable,
                "pooling": pooling,
                "layer": layer,
                "lr": lr,
                "wd": wd,
                "epochs": ep,
                "val_mse": vmse,
                "test_r2": m["r2"],
                "test_mae": m["mae"],  # circular MAE in degrees for direction
            }
        )

        # Optional paper-style CV scores
        if cfg.paper_cv:
            rows[-1].update(_paper_cv(X, Ttr, ytr, cv_splits, cfg, is_dir, zt))
            print(
                f"   paper-cv: r2={rows[-1]['cv_r2_mean']:.3f}+-{rows[-1]['cv_r2_std']:.3f} "
                f"mae={rows[-1]['cv_mae_mean']:.3f}+-{rows[-1]['cv_mae_std']:.3f}"
            )
        print(
            f"{variable}/{pooling} L{layer:02d} lr={lr:g} wd={wd:g} ep={ep} "
            f"val_mse={vmse:.4f} test_r2={m['r2']:.3f} test_mae={m['mae']:.3f}"
        )

        # Save after every layer so a crash loses at most one layer
        if csv_path is not None:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    return pd.DataFrame(rows)