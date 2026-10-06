import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.metrics import deg_to_sincos
from src.probe import _metrics, load_split_data

ROOT = Path(__file__).resolve().parents[1]

# Paper hparams (acceleration not given in the paper; uses speed's)
PAPER_HP = {
    "direction": dict(opt="adam", lr=1e-3, wd=1e-4, epochs=100),
    "speed": dict(opt="adam", lr=1e-3, wd=1e-4, epochs=50),
    "acceleration": dict(opt="adam", lr=1e-3, wd=1e-4, epochs=50),
}


def get_hparams(mode, variable, pooling, layer, run_name, results_dir=ROOT / "results") -> dict:
    # 'paper': fixed paper hparams; 'item1': lr / wd / epochs selected for this layer in Item 1
    if mode == "paper":
        return dict(PAPER_HP[variable])
    csv = Path(results_dir) / run_name / f"probes_{variable}_{pooling}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"--hp item1 needs {csv} (run Item 1 first)")
    df = pd.read_csv(csv)
    r = df[df["layer"] == layer]
    if r.empty:
        raise ValueError(f"layer {layer} not in {csv.name}")
    r = r.iloc[0]
    return dict(opt="adamw", lr=float(r["lr"]), wd=float(r["wd"]), epochs=int(r["epochs"]))


def _fit(X, T, hp, batch_size, seed, device):
    # Linear probe, MSE, fixed hparams and fixed number of epochs
    torch.manual_seed(seed)
    model = torch.nn.Linear(X.shape[1], T.shape[1]).to(device)
    Opt = torch.optim.Adam if hp["opt"] == "adam" else torch.optim.AdamW
    opt = Opt(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    n = len(X)
    bs = batch_size or n
    for _ in range(hp["epochs"]):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            loss = F.mse_loss(model(X[idx]), T[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


def _acc_within(y_deg, pred_sincos, tol=15.0):
    # % of samples whose decoded angle is within +-tol deg of the truth (circular)
    ang = np.degrees(np.arctan2(pred_sincos[:, 0], pred_sincos[:, 1]))
    err = np.abs((ang - np.asarray(y_deg) + 180.0) % 360.0 - 180.0)
    return float((err <= tol).mean() * 100.0)


def _at_chance(m, is_dir, mae_base, stop_on="any"):
    # Paper chance thresholds; stop_on: 'r2' | 'mae' | 'any' (paper: either)
    r2_bad = m["r2"] < (0.1 if is_dir else 0.05)
    mae_bad = m["mae"] > (80.0 if is_dir else 0.9 * mae_base)
    return {"r2": r2_bad, "mae": mae_bad, "any": r2_bad or mae_bad}[stop_on]


def _stop_rule(is_dir, mae_base, stop_on):
    # Readable version of the stop rule, for the log
    r2 = "test R² < 0.1" if is_dir else "test R² < 0.05"
    mae = "circular MAE > 80°" if is_dir else f"MAE > 0.9 x mean-baseline ({0.9 * mae_base:.3f})"
    return {"r2": r2, "mae": mae, "any": f"{r2} or {mae}"}[stop_on]


def _stem(variable, pooling, layer, hp_mode, stop_on, dir_remove):
    # Output name; non-default stop rule / removal mode are added as tags
    stem = f"{variable}_{pooling}_L{layer:02d}_{hp_mode}"
    if stop_on != "any":
        stem += f"_stop-{stop_on}"
    if dir_remove != "both":
        stem += f"_rm-{dir_remove}"
    return stem


def _summary(variable, layer, hp_mode, stop_on, dir_remove, dim, rm_dim, baseline_r2, stop):
    return {"variable": variable, "layer": layer, "hp": hp_mode, "stop_on": stop_on,
            "dir_remove": dir_remove, "dim": dim, "probes": dim // rm_dim,
            "baseline_r2": baseline_r2, "stop": stop}


def run_nullspace(
    variable: str,
    pooling: str,
    layer: int,
    hp_mode: str,
    run_name: str = "paper",
    max_iters: int = 200,
    batch_size: int = 128,
    seed: int = 0,
    device: str = "cuda",
    stop_on: str = "any",
    dir_remove: str = "both",
    resume: bool = False,
    out_subdir: str = "nullspace",
    results_dir: Path = ROOT / "results",
) -> dict:
    # dir_remove (direction only): project out 'both' rows (sin + cos, 2 dims), or only 'sin' / 'cos' (1 dim);
    # the probe is always trained on [sin, cos]
    if dir_remove not in ("both", "sin", "cos"):
        raise ValueError(f"dir_remove must be both|sin|cos, got {dir_remove}")

    # Data for this layer; targets (sin, cos) for direction, scalar otherwise
    Xtr_all, ytr, Xte_all, yte = load_split_data(variable, pooling)
    is_dir = variable == "direction"
    f32 = dict(dtype=torch.float32, device=device)
    Xtr = torch.tensor(Xtr_all[:, layer], **f32)
    Xte = torch.tensor(Xte_all[:, layer], **f32)
    Ttr = torch.tensor(deg_to_sincos(ytr) if is_dir else ytr[:, None], **f32)
    hp = get_hparams(hp_mode, variable, pooling, layer, run_name, results_dir)
    d, out_dim = Xtr.shape[1], Ttr.shape[1]

    # Probe rows to remove each iteration (scalar probes have a single row)
    if not is_dir:
        dir_remove = "both"
    rm_rows = {"both": list(range(out_dim)), "sin": [0], "cos": [1]}[dir_remove]
    rm_dim = len(rm_rows)
    mae_base = None if is_dir else float(np.abs(yte - ytr.mean()).mean())  # predict-train-mean baseline
    unit = "°" if is_dir else ""

    # Output paths
    out_dir = Path(results_dir) / run_name / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _stem(variable, pooling, layer, hp_mode, stop_on, dir_remove)
    csv_path, npz_path = out_dir / f"{stem}.csv", out_dir / f"{stem}_basis.npz"

    # Resume: npz is written only when a run finishes, so skip it
    if resume and npz_path.exists() and csv_path.exists():
        prev = pd.read_csv(csv_path)
        dim = int(np.load(npz_path)["V"].shape[1])
        last = prev.iloc[-1]
        print(f"[resume] skip {stem}: done ({dim} dims)")
        stop = f"probe {int(last['iter'])} at chance" if bool(last["at_chance"]) else "max_iters (resumed)"
        return _summary(variable, layer, hp_mode, stop_on, dir_remove, dim, rm_dim,
                        float(prev.iloc[0]["test_r2"]), stop)

    line = "=" * 88
    print(f"\n{line}\nNullspace probing | variable={variable}  pooling={pooling}  layer={layer}  hp={hp_mode}")
    print(f"  hparams : {hp['opt']} lr={hp['lr']:g} wd={hp['wd']:g} epochs={hp['epochs']} batch={batch_size or 'full'}")
    print(f"  data    : train N={len(Xtr)}  test N={len(Xte)}  d={d}  dims removed per probe={rm_dim}"
          + (f"  (removing: {dir_remove})" if is_dir else ""))
    print(f"  stop    : {_stop_rule(is_dir, mae_base, stop_on)}  (max {max_iters} probes)\n{line}")

    V = torch.zeros(d, 0, **f32)  # removed basis (orthonormal columns)
    Ws, bs = [], []               # removed probes: orthogonalized weight [out_dim, d] and bias [out_dim]
    rows, prev_r2 = [], None
    stop_reason = f"reached max_iters={max_iters}"
    bar = tqdm(range(1, max_iters + 1), desc=f"{variable} L{layer} [{hp_mode}]", unit="probe", dynamic_ncols=True)

    for k in bar:
        t0 = time.time()

        # Fit a probe on the current activations, evaluate on test
        model = _fit(Xtr, Ttr, hp, batch_size, seed, device)
        with torch.no_grad():
            m_tr = _metrics(ytr, model(Xtr).cpu().numpy(), is_dir)
            p_te = model(Xte).cpu().numpy()
            m_te = _metrics(yte, p_te, is_dir)
        acc15 = _acc_within(yte, p_te, 15.0) if is_dir else float("nan")
        removed = V.shape[1]
        chance = _at_chance(m_te, is_dir, mae_base, stop_on)

        # Log and save every iteration
        rows.append({
            "iter": k, "dims_removed": removed, "test_r2": m_te["r2"], "test_mae": m_te["mae"],
            "test_acc15": acc15, "train_r2": m_tr["r2"], "at_chance": chance, **hp,
        })
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        delta = "      " if prev_r2 is None else f" ({m_te['r2'] - prev_r2:+.3f})"
        tqdm.write(
            f"  [k={k:03d}] Orthogonal Probe Number={removed:4d} | test R²={m_te['r2']:6.3f}{delta} | "
            f"test MAE={m_te['mae']:7.3f}{unit}"
            + (f" | acc@15°={acc15:5.1f}%" if is_dir else "")
            + f" | train R²={m_tr['r2']:6.3f} | {time.time() - t0:4.1f}s"
            + ("  <- at chance" if chance else "")
        )
        bar.set_postfix(R2=f"{m_te['r2']:.3f}", MAE=f"{m_te['mae']:.2f}", dims=removed)
        if chance:
            stop_reason = f"probe {k} at chance"
            break

        # Orthogonalize W against the removed basis, then project its subspace out of both splits
        W_full = model.weight.detach().clone()        # [out_dim, d]
        W_full = W_full - (W_full @ V) @ V.T
        Ws.append(W_full.cpu().numpy())
        bs.append(model.bias.detach().cpu().numpy())
        Q, _ = torch.linalg.qr(W_full[rm_rows].T)     # [d, rm_dim], orthonormal
        V = torch.cat([V, Q], dim=1)
        Xtr = Xtr - (Xtr @ Q) @ Q.T
        Xte = Xte - (Xte @ Q) @ Q.T
        prev_r2 = m_te["r2"]
    bar.close()

    # Sanity: V orthonormal, and no train activation left along V
    dim = V.shape[1]
    with torch.no_grad():
        orth_err = float(torch.linalg.norm(V.T @ V - torch.eye(dim, **f32))) if dim else 0.0
        leak = float(torch.linalg.norm(Xtr @ V) / torch.linalg.norm(Xtr)) if dim else 0.0

    # Save removed basis and probes
    np.savez(
        npz_path, V=V.cpu().numpy(),
        W=np.stack(Ws) if Ws else np.zeros((0, out_dim, d), np.float32),  # [K, out_dim, d]
        b=np.stack(bs) if bs else np.zeros((0, out_dim), np.float32),     # [K, out_dim]
        variable=variable, pooling=pooling, layer=layer,
        hp=json.dumps(hp), hp_mode=hp_mode, dims_per_probe=rm_dim, stop_on=stop_on,
        dir_remove=dir_remove,
    )
    print(f"{line}\n  stopped : {stop_reason}")
    print(f"  result  : subspace dimensionality = {dim} dims ({dim // rm_dim} probes removed)")
    print(f"  baseline: test R²={rows[0]['test_r2']:.3f} -> final test R²={rows[-1]['test_r2']:.3f}")
    print(f"  check   : ||VᵀV - I|| = {orth_err:.2e}   ||X_tr V|| / ||X_tr|| = {leak:.2e}")
    print(f"  saved   : {csv_path}\n            {npz_path}\n{line}")
    return _summary(variable, layer, hp_mode, stop_on, dir_remove, dim, rm_dim, rows[0]["test_r2"], stop_reason)