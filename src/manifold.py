from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA

PERIOD = 360.0
CYCLIC = {"direction"}  # variables with a periodic spline


@dataclass
class Manifold:
    variable: str
    layer: int
    periodic: bool
    mean: np.ndarray          # [d]   PCA mean
    comps: np.ndarray         # [k, d] PCA components (rows orthonormal)
    evr: np.ndarray           # [k]   explained variance ratio of the PCA (train)
    u: np.ndarray             # [m]   label coordinate of each centroid
    C: np.ndarray             # [m, k] centroids in PCA coordinates
    counts: np.ndarray        # [m]   train samples per centroid
    held_out_u: np.ndarray = field(default_factory=lambda: np.zeros(0))  # bin coords left out of the fit
    n_bins: int = 64
    spline: CubicSpline = None

    def to_pca(self, X):
        # Activations -> PCA coordinates
        return (X - self.mean) @ self.comps.T

    def s(self, u):
        # Spline point(s) at label coordinate u (wrapped to [0, 360) for direction)
        u = np.asarray(u, dtype=np.float64)
        return self.spline(np.mod(u, PERIOD) if self.periodic else u)

    def save(self, path: Path):
        np.savez(path, variable=self.variable, layer=self.layer, periodic=self.periodic, mean=self.mean,
                 comps=self.comps, evr=self.evr, u=self.u, C=self.C, counts=self.counts,
                 held_out_u=self.held_out_u, n_bins=self.n_bins)

    @classmethod
    def load(cls, path: Path):
        # Load saved fields and rebuild the spline
        z = np.load(path, allow_pickle=False)
        m = cls(variable=str(z["variable"]), layer=int(z["layer"]), periodic=bool(z["periodic"]),
                mean=z["mean"], comps=z["comps"], evr=z["evr"], u=z["u"], C=z["C"], counts=z["counts"],
                held_out_u=z["held_out_u"], n_bins=int(z["n_bins"]))
        m.spline = make_spline(m.u, m.C, m.periodic)
        return m


def make_spline(u, C, periodic):
    # Cubic spline through the centroids: periodic for direction, natural for scalars
    if periodic:
        # Close the loop: repeat the first knot one period later
        return CubicSpline(np.append(u, u[0] + PERIOD), np.vstack([C, C[:1]]), bc_type="periodic",
                           extrapolate="periodic")
    return CubicSpline(u, C, bc_type="natural")


def bin_index(y, labels, n_bins):
    # Bin index per sample, grouping adjacent label values by rank (labels = sorted unique values)
    if len(labels) % n_bins:
        raise ValueError(f"n_bins={n_bins} must divide the number of label values ({len(labels)})")
    rank = np.searchsorted(labels, y)
    return rank // (len(labels) // n_bins)


def fit_manifold(variable, layer, Xtr, ytr, labels, pca_dim=64, n_bins=64, drop_every=0, seed=0) -> Manifold:
    # Fit on train only: PCA -> per-bin centroids -> spline through the kept centroids
    periodic = variable in CYCLIC
    pca = PCA(n_components=pca_dim, random_state=seed).fit(Xtr)
    Z = pca.transform(Xtr)

    # Centroid per bin: mean PCA coords; its coordinate u = mean label of the bin
    b = bin_index(ytr, labels, n_bins)
    per = len(labels) // n_bins
    u_all = np.array([labels[i * per:(i + 1) * per].mean() for i in range(n_bins)])
    C_all = np.stack([Z[b == i].mean(0) for i in range(n_bins)])
    cnt_all = np.array([(b == i).sum() for i in range(n_bins)])

    # Optionally hold out bins 1, 1+k, 1+2k, ... (scalars: never the two ends)
    keep = np.ones(n_bins, bool)
    if drop_every and drop_every > 1:
        idx = np.arange(1, n_bins, drop_every)
        if not periodic:
            idx = idx[idx < n_bins - 1]
        keep[idx] = False

    m = Manifold(variable=variable, layer=layer, periodic=periodic, mean=pca.mean_, comps=pca.components_,
                 evr=pca.explained_variance_ratio_, u=u_all[keep], C=C_all[keep], counts=cnt_all[keep],
                 held_out_u=u_all[~keep], n_bins=n_bins)
    m.spline = make_spline(m.u, m.C, periodic)
    return m


def short_delta(u0, u1, periodic):
    # Signed change from u0 to u1; the short way around the circle if periodic
    d = u1 - u0
    return ((d + PERIOD / 2) % PERIOD) - PERIOD / 2 if periodic else d


def path_u(u0, u1, T, periodic):
    # T+1 label coordinates from u0 to u1 (short arc for direction)
    return u0 + np.linspace(0, 1, T + 1) * short_delta(u0, u1, periodic)