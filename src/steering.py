from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeCV

from src.metrics import deg_to_sincos, direction_metrics, scalar_metrics
from src.nullspace import PAPER_HP, _fit
from src.probe import load_split_data

UNIT = {"direction": "°", "speed": "", "acceleration": ""}


def circ_err(a, b):
    # Circular error in degrees
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


def decode_deg(p):
    # (sin, cos) -> angle in [0, 360)
    return np.degrees(np.arctan2(p[:, 0], p[:, 1])) % 360.0


def encode(variable, y):
    # Labels -> probe targets: [n, 2] (sin, cos) for direction, [n, 1] for scalars
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    return deg_to_sincos(y).astype(np.float64) if variable == "direction" else y[:, None]


def decode(variable, p):
    # Probe outputs -> label units
    return decode_deg(p) if variable == "direction" else p[:, 0]


def err(variable, a, b):
    return circ_err(a, b) if variable == "direction" else np.abs(np.asarray(a) - np.asarray(b))


def load_probes(basis_dir: Path, variable: str, pooling: str, layer: int):
    # Steering probes saved by a nullspace run (paper hp): W [K, out, d], b [K, out]
    z = np.load(Path(basis_dir) / f"{variable}_{pooling}_L{layer:02d}_paper_basis.npz")
    return z["W"].astype(np.float64), z["b"].astype(np.float64)


def steer(X, W, b, t):
    # Set every probe's output to target t inside span(V), keep the orthogonal part of x:
    # x* = (x - V Vᵀ x) + V c*, with (M V) c* = t - b stacked over the K probes
    K, out, d = W.shape
    M = W.reshape(K * out, d)                    # stacked probe rows
    V, _ = np.linalg.qr(M.T)                     # [d, K*out], orthonormal
    A = M @ V                                    # [K*out, K*out]
    rhs = np.tile(t, K) - b.reshape(-1)
    c, *_ = np.linalg.lstsq(A, rhs, rcond=None)  # target coordinates (same for every x)
    Xs = X - (X @ V) @ V.T + c @ V.T

    # Sanity: max |probe(x*) - t| over all probes
    resid = float(np.abs(Xs @ M.T + b.reshape(-1) - np.tile(t, K)).max())
    return Xs, resid


def train_eval_probe(X, y, variable, mode, seed, device):
    # Fresh probe on unsteered test activations only:
    # 'sgd' = linear probe with paper hp (init seed = seed); 'ridge' = RidgeCV, alpha by 5-fold CV
    T = encode(variable, y)
    if mode == "sgd":
        Tt = torch.tensor(T, dtype=torch.float32, device=device)
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        model = _fit(Xt, Tt, PAPER_HP[variable], 128, seed, device)
        W = model.weight.detach().cpu().numpy().astype(np.float64)
        b = model.bias.detach().cpu().numpy().astype(np.float64)
        info = f"sgd (paper hp, seed={seed})"
    elif mode == "ridge":
        r = RidgeCV(alphas=np.logspace(-2, 5, 15), cv=5).fit(X, T)
        W, b = np.atleast_2d(r.coef_).astype(np.float64), np.atleast_1d(r.intercept_).astype(np.float64)
        info = f"ridge (alpha={r.alpha_:g})"
    else:
        raise ValueError(mode)

    P = X @ W.T + b
    m = direction_metrics(y, P) if variable == "direction" else scalar_metrics(y, P[:, 0])
    return W, b, m, info


def _default_target(variable, labels, y_all):
    # 90° for direction; for scalars the smallest label >= the label mean (mean falls between two labels)
    if variable == "direction":
        return 90.0
    return float(labels[labels >= y_all.mean() - 1e-9][0])


def _k_values(K_max, k_step):
    # K = 0 (unsteered), 1, 1+step, ..., always including K_max
    ks = [0] + list(range(1, K_max + 1, k_step))
    if (K_max - 1) % k_step:
        ks.append(K_max)
    return ks


def _eval_in_span(We, W, K):
    # Fraction of the eval probe's weight norm inside the steering subspace of the first K probes
    if K == 0:
        return 0.0
    VK, _ = np.linalg.qr(W[:K].reshape(-1, W.shape[-1]).T)
    return float(np.linalg.norm(We @ VK) / np.linalg.norm(We))


def run_steering(variable, pooling, layer, basis_dir, target=None, avg_targets=False, k_step=1, seed=0,
                 device="cpu", eval_probe="ridge", eval_seed=0):
    # Test activations for this layer, all label values
    _, ytr, Xte_all, yte = load_split_data(variable, pooling)
    X = Xte_all[:, layer].astype(np.float64)
    y_all = np.concatenate([ytr, yte])
    labels = np.sort(np.unique(y_all))

    # Main target, plus every other label value if averaging
    target = float(_default_target(variable, labels, y_all) if target is None else target)
    targets = [target] + ([float(t) for t in labels if not np.isclose(t, target)] if avg_targets else [])
    u = UNIT[variable]

    # Eval probe (test only) and steering probes (train, from the nullspace run)
    We, be, m_eval, eval_info = train_eval_probe(X, yte, variable, eval_probe, eval_seed, device)
    W, b = load_probes(basis_dir, variable, pooling, layer)
    K_max, out = W.shape[0], W.shape[1]

    def read(Z):
        return decode(variable, Z @ We.T + be)

    line = "=" * 100
    print(f"\n{line}\nSteering | {variable}  pooling={pooling}  layer={layer}  K_max={K_max} "
          f"({out * K_max} dims)\n  eval probe (test only, {eval_info}): R²={m_eval['r2']:.3f}  MAE={m_eval['mae']:.3f}{u}"
          f"\n  target={target:g}{u}" + (f"  + avg over {len(targets)} targets (all label values)" if avg_targets else "")
          + ("   (not an exact label value)" if not np.isclose(labels, target).any() else "")
          + f"\n  test N={len(X)}  label range [{labels.min():g}, {labels.max():g}]\n{line}")
    hdr = f"{'K':>4} {'dims':>5} | {'target=' + format(target, 'g') + u + ': →target':>22} {'→truth':>8}"
    if avg_targets:
        hdr += f" | {'avg: →target':>13} {'→truth':>8}"
    print(hdr + f" | {'edit ‖Δ‖/‖x‖':>13} | {'eval-in-span':>12} | {'max probe resid':>15}")

    rows = []
    for K in _k_values(K_max, k_step):
        overlap = _eval_in_span(We, W, K)

        # Steer to each target with the first K probes, read out with the eval probe
        per_t = []
        for t in targets:
            Xs, resid = (X, 0.0) if K == 0 else steer(X, W[:K], b[:K], encode(variable, t)[0])
            pred = read(Xs)
            per_t.append(dict(
                variable=variable, pooling=pooling, layer=layer, K=K, dims=out * K, target=t,
                mae_to_target=float(err(variable, pred, t).mean()),
                mae_to_truth=float(err(variable, pred, yte).mean()),
                acc15_target=float((circ_err(pred, t) <= 15).mean() * 100) if variable == "direction" else np.nan,
                edit_rel=float(np.linalg.norm(Xs - X) / np.linalg.norm(X)), probe_resid=resid,
                eval_r2=m_eval["r2"], eval_mae=m_eval["mae"], eval_in_span=overlap,
                eval_probe=eval_probe, eval_seed=eval_seed,
            ))
        rows += per_t

        # Table row: main target, optional average over targets
        main = per_t[0]
        s = f"{K:>4} {out * K:>5} | {main['mae_to_target']:>21.3f}{u or ' '} {main['mae_to_truth']:>7.3f}{u or ' '}"
        if avg_targets:
            s += (f" | {np.mean([r['mae_to_target'] for r in per_t]):>12.3f}{u or ' '}"
                  f" {np.mean([r['mae_to_truth'] for r in per_t]):>7.3f}{u or ' '}")
        print(s + f" | {main['edit_rel']:>13.3f} | {overlap:>12.3f} | {max(r['probe_resid'] for r in per_t):>15.1e}")
    return rows