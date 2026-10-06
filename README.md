# Interpreting Physics in V-JEPA

An interpretability study of how a frozen **V-JEPA 2** [3] video encoder (ViT-L, `facebook/vjepa2-vitl-fpc64-256`) represents three physical properties of simple moving objects: **direction**, **speed**, and **acceleration**.

- **Part 1** reproduces the main experimental progression of *Interpreting Physics in Video World Models* [1]: layer-wise probing, iterative nullspace probing, and multi-probe subspace steering.
- **Part 2** extends it with the manifold (spline) steering of *Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior* [2], and compares it with linear (chord) and multi-probe steering.

## 📦 What's in this Repo

**Part 1: reproducing the physics results**

- Feature extraction: one pass of frozen V-JEPA 2 per clip, caching mean-pooled and last-time-slot-pooled activations for all 25 layers
- Stratified 75/25 train/test split per dataset, with hyperparameters selected only inside train
- Layer-wise linear probes (`nn.Linear`, AdamW, MSE) over the paper's lr × wd grid, with paper-style cross-validation; direction is regressed as (sin θ, cos θ)
- Iterative nullspace probing: fit a probe, project its readout subspace out of the activations, refit, until the probe reaches chance
- Multi-probe subspace steering: set the outputs of K stacked probes to a target value, evaluated with a fresh probe trained only on held-out test activations

**Part 2: manifold steering**

- PCA (64 dims, train only) + label-binned centroids + cubic spline (periodic for direction, natural for speed / acceleration)
- Steering along start → target paths with three methods: spline (on the manifold), chord (straight line between endpoint centroids), and multi-probe (Part 1)
- Held-out-bin test: bins left out of the spline fit, comparing spline, linear interpolation of the neighbouring centroids, and multi-probe
- Plots of the manifold geometry, the steered paths, and per-method summaries

## Repository structure

```
src/
  data.py         # load manifests + metadata into a dataframe; dataset sanity report
  metrics.py      # R² / MAE, circular MAE, (sin, cos) <-> degrees
  probe.py        # ProbeConfig, cached-feature loading, per-layer probe training + selection
  nullspace.py    # iterative nullspace probing (orthogonalized probe removal)
  steering.py     # multi-probe subspace steering + held-out evaluation probe
  manifold.py     # PCA + centroid spline manifolds, path helpers

scripts/
  01_make_splits.py          # stratified train/test split per variable
  02_extract_features.py     # run V-JEPA 2 once per clip, cache pooled activations
  03_run_probes.py           # Part 1, item 1: layer-wise probing
  04_plot_layers.py          #   probe score vs layer
  05_nullspace.py            # Part 1, item 2: iterative nullspace probing
  06_plot_nullspace.py       #   nullspace curves + probes-until-chance vs layer
  07_summarize_nullspace.py  #   per-layer table: probe scores + nullspace dims
  08_steer.py                # Part 1, item 3: multi-probe subspace steering (K sweep)
  09_plot_steering.py        #   MAE vs number of steering probes
  10_manifold_fit.py         # Part 2: fit PCA + spline manifolds, geometry plots
  11_manifold_steer.py       #   spline vs chord vs multi-probe steering
  12_manifold_plots.py       #   path plots, summaries, held-out bins
```

## Setup

Tested with Python 3.10 on a single GPU.

```bash
git clone <this-repo>
cd vjepa-assignment
conda create -n vjepa python=3.10 -y
conda activate vjepa
pip install torch transformers av numpy pandas scikit-learn scipy matplotlib tqdm
```

Run every command from the repo root. The datasets are read from `../data` (relative to the working directory), with one folder per variable:

```
data/
├── direction/     manifest.jsonl + videos/scene_XXXX/{video.mp4, metadata.json}
├── speed/
└── acceleration/
```

Check that everything loads (clip counts, missing videos, label ranges):

```bash
python src/data.py                 # all three variables
python src/data.py speed           # one variable
```

Outputs go to `cache/` (splits, features) and `results/<run-name>/` (CSVs, saved bases, manifolds, figures). Most scripts take `--results-dir` and `--run-name` (default `paper`).

## Running the pipeline

### 1. Splits and features

```bash
python scripts/01_make_splits.py
python scripts/02_extract_features.py
```

`01` writes `cache/splits_{variable}.json` (stratified on the 64 label values, seed 0). `02` writes `cache/feats_{variable}.npz` with `feats_mean` (mean over all 2048 tokens) and `feats_last` (mean over space of the last time slot), each `[N, 25, 1024]`. Layer 0 is the patch embedding; layers 1–24 are the transformer blocks.

Quick test of the extraction on a few clips (nothing is saved):

```bash
python scripts/02_extract_features.py speed --limit 16
```

| Arg | Default | Meaning |
|---|---|---|
| `--test-size` (01) | `0.25` | Test fraction |
| `--model-id` (02) | `facebook/vjepa2-vitl-fpc64-256` | Hugging Face model |
| `--batch-size` (02) | `8` | Clips per forward pass |
| `--cache-dir` | `cache` | Where splits / features are written |

### 2. Layer-wise probing (Part 1, item 1)

```bash
python scripts/03_run_probes.py --run-name paper --paper-cv
python scripts/04_plot_layers.py --run-name paper --mark-layer 9 --show-test
```

For each layer, `(lr, wd, epochs)` is chosen on a stratified inner validation split of train, the probe is refit on all of train, and it is evaluated once on test. `--paper-cv` adds paper-style cross-validation scores (mean ± std over folds), which `04` plots.

| Arg | Default | Meaning |
|---|---|---|
| `--poolings` | `mean last` | Which cached pooling(s) to probe |
| `--lrs` / `--wds` | paper grid | Learning rates / weight decays to search |
| `--cv-folds` | `0` | `0` = single inner val split, `k` = stratified k-fold inside train |
| `--standardize-features` / `--standardize-targets` | off | Optional z-scoring (off = paper) |
| `--save-probes` | off | Save the final probe per layer to `checkpoints/` |
| `--resume` | off | Skip layers already in the run's CSVs |

### 3. Iterative nullspace probing (Part 1, item 2)

```bash
for L in $(seq 1 24); do python scripts/05_nullspace.py --layers $L; done
python scripts/06_plot_nullspace.py --layer-summary
python scripts/07_summarize_nullspace.py
```

Each iteration fits a probe with the paper's fixed hyperparameters, stops if test performance is at chance (paper thresholds), and otherwise orthogonalizes the probe weights against the already-removed basis and projects that subspace out of both splits. Results go to `results/paper/nullspace/`: a per-iteration CSV and a `_basis.npz` with the removed basis and probe weights. `07` joins the probe scores with the nullspace dimensionality per layer.

| Arg | Default | Meaning |
|---|---|---|
| `--layers` | required | One layer per variable, or one layer for all |
| `--max-iters` | `200` | Max probes per run |
| `--stop-on` | `any` | Chance criterion: `r2`, `mae`, or either (paper) |
| `--dir-remove` | `both` | Direction only: remove both (sin, cos) rows, or only one |
| `--out-subdir` | `nullspace` | Output folder under `results/<run-name>/` |

### 4. Multi-probe subspace steering (Part 1, item 3)

```bash
python scripts/08_steer.py --variables direction speed acceleration --layers 8 9 15 20 22
python scripts/09_plot_steering.py
```

The steering probes are the probe weights saved by `05`, read from `results/<run-name>/<steer-dir>/` (default `steer_orth`; run `05` with `--out-subdir steer_orth` for the steering layers, or pass `--steer-dir nullspace`). For each K, activations are edited so the first K probes all read the target, and the result is read out with a ridge probe trained only on unsteered test activations.

| Arg | Default | Meaning |
|---|---|---|
| `--target` | auto | 90° for direction; for speed / acceleration the label just above the mean |
| `--avg-targets` | off | Also steer to every label value and average |
| `--k-step` | `1` | Sweep K = 1, 1+step, … (K_max always included) |
| `--eval-probe` | `ridge` | Held-out evaluation probe: `ridge` and/or `sgd` |

### 5. Manifold steering (Part 2)

```bash
python scripts/10_manifold_fit.py
python scripts/10_manifold_fit.py --n-bins 32 --drop-every 4

python scripts/11_manifold_steer.py
python scripts/11_manifold_steer.py --drop-every 4 --heldout

python scripts/12_manifold_plots.py
```

`10` fits PCA + centroids + spline on train and plots the geometry (data-PCA and centroid-PCA views, explained variance, curve per layer). `11` steers test activations along start → target paths (direction `0:180` and `0:90`, scalars `min:max`) with spline, chord, and multi-probe, and records the read value, error, off-manifold distance, edit size, and (for direction) readout confidence at every step. With `--drop-every 4 --heldout`, it also steers straight to bins left out of the spline fit. `12` makes the path, summary, and held-out figures.

| Arg | Default | Meaning |
|---|---|---|
| `--layers` | `8 9 15 20 22` | Layers to fit / steer |
| `--n-bins` | `64 32` (10), `32` (11) | Label bins per centroid (must divide the 64 label values) |
| `--drop-every` | `0` | Hold out every k-th bin from the spline fit |
| `--pairs` (11) | per variable | Paths as `start:target`, e.g. `0:180 0:90` |
| `--T` (11) | `20` | Steps along each path |
| `--edit` (11) | `replace` | `replace` the PCA coordinates, or `shift` them by target − start |
| `--mp-k` (11) | `0` | Probes for multi-probe steering (`0` = all) |

## References

[1] Joseph, S., Garrido, Q., Balestriero, R., Kowal, M., Fel, T., Bakhtiari, S., Richards, B., & Rabbat, M. (2026). Interpreting physics in video world models. *arXiv preprint arXiv:2602.07050*.

[2] Wurgaft, D., Rager, C., Kowal, M., Shyam, V., Feucht, S., Bhalla, U., Haklay, T., Bigelow, E., Sarfati, R., McGrath, T., Lewis, O., Merullo, J., Goodman, N. D., Fel, T., Geiger, A., & Lubana, E. S. (2026). Manifold steering reveals the shared geometry of neural network representation and behavior. *arXiv preprint arXiv:2605.05115*.

[3] Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., Komeili, M., Muckley, M., Rizvi, A., Roberts, C., Sinha, K., Zholus, A., Arnaud, S., Gejji, A., Martin, A., Robert Hogan, F., Dugas, D., Bojanowski, P., Khalidov, V., Labatut, P., Massa, F., Szafraniec, M., Krishnakumar, K., Li, Y., Ma, X., Chandar, S., Meier, F., LeCun, Y., Rabbat, M., & Ballas, N. (2025). V-JEPA 2: Self-supervised video models enable understanding, prediction and planning. *arXiv preprint arXiv:2506.09985*.

```bibtex
@article{joseph2026interpreting,
  title={Interpreting Physics in Video World Models},
  author={Joseph, Sonia and Garrido, Quentin and Balestriero, Randall and Kowal, Matthew and Fel, Thomas and Bakhtiari, Shahab and Richards, Blake and Rabbat, Mike},
  journal={arXiv preprint arXiv:2602.07050},
  year={2026}
}

@article{wurgaft2026manifold,
  title={Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior},
  author={Wurgaft, Daniel and Rager, Can and Kowal, Matthew and Shyam, Vasudev and Feucht, Sheridan and Bhalla, Usha and Haklay, Tal and Bigelow, Eric and Sarfati, Raphael and McGrath, Thomas and Lewis, Owen and Merullo, Jack and Goodman, Noah D. and Fel, Thomas and Geiger, Atticus and Lubana, Ekdeep Singh},
  journal={arXiv preprint arXiv:2605.05115},
  year={2026}
}

@article{assran2025vjepa2,
  title={V-JEPA~2: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
  author={Assran, Mahmoud and Bardes, Adrien and Fan, David and Garrido, Quentin and Howes, Russell and Komeili, Mojtaba and Muckley, Matthew and Rizvi, Ammar and Roberts, Claire and Sinha, Koustuv and Zholus, Artem and Arnaud, Sergio and Gejji, Abha and Martin, Ada and Robert Hogan, Francois and Dugas, Daniel and Bojanowski, Piotr and Khalidov, Vasil and Labatut, Patrick and Massa, Francisco and Szafraniec, Marc and Krishnakumar, Kapil and Li, Yong and Ma, Xiaodong and Chandar, Sarath and Meier, Franziska and LeCun, Yann and Rabbat, Michael and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2506.09985},
  year={2025}
}
```

## Acknowledgement

This work uses the pretrained [V-JEPA 2](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) [3] encoder via Hugging Face Transformers, and is built with [PyTorch](https://pytorch.org/), [scikit-learn](https://scikit-learn.org/), [SciPy](https://scipy.org/), and [Matplotlib](https://matplotlib.org/). We thank the authors of [1], [2], and [3] for the methods and model this project builds on.
