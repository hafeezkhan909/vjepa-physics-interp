import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def sincos_to_deg(pred: np.ndarray) -> np.ndarray:
    # pred[:, 0] = sin, pred[:, 1] = cos -> angle in [0, 360)
    return np.degrees(np.arctan2(pred[:, 0], pred[:, 1])) % 360.0


def circ_mae_deg(true_deg: np.ndarray, pred_deg: np.ndarray) -> float:
    # MAE in degrees, taking the shorter way around the circle
    d = np.abs(true_deg - pred_deg) % 360.0
    return float(np.minimum(d, 360.0 - d).mean())


def deg_to_sincos(deg: np.ndarray) -> np.ndarray:
    t = np.radians(deg)
    return np.stack([np.sin(t), np.cos(t)], axis=1)


def scalar_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    return {"r2": float(r2_score(y, pred)), "mae": float(mean_absolute_error(y, pred))}


def direction_metrics(theta_deg: np.ndarray, pred_sincos: np.ndarray) -> dict:
    # R² on the (sin, cos) targets (mean of the two); MAE = circular MAE in degrees
    return {
        "r2": float(r2_score(deg_to_sincos(theta_deg), pred_sincos)),
        "mae": circ_mae_deg(theta_deg, sincos_to_deg(pred_sincos)),
    }