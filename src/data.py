import argparse
import json
from pathlib import Path

import pandas as pd

DATA_ROOT = Path("../data")
VARIABLES = ["direction", "speed", "acceleration"]
# Metadata column used as the label for each variable
TARGET_COL = {
    "direction": "theta_degrees",
    "speed": "speed_mps",
    "acceleration": "acceleration_mps2",
}


def load_dataset(variable: str, data_root: Path = DATA_ROOT) -> pd.DataFrame:
    # One row per clip: id, video_path, all metadata fields, and a `target` column
    root = Path(data_root) / variable
    rows = []
    with open(root / "manifest.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            m = json.loads(line)
            with open(root / m["metadata"]) as mf:
                meta = json.load(mf)
            meta.pop("id", None)  # manifest id is the source of truth
            x, y = meta.pop("start_position_xy_m")
            rows.append(
                {
                    "id": m["id"],
                    "video_path": str(root / m["video"]),
                    **meta,
                    "start_x": x,
                    "start_y": y,
                }
            )
    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    df["target"] = df[TARGET_COL[variable]]
    return df


def inspect(variable: str, df: pd.DataFrame) -> None:
    # Sanity report for one dataset
    print(f"\n===== {variable} =====")
    print(f"clips: {len(df)}")

    missing = [p for p in df["video_path"] if not Path(p).exists()]
    print(f"missing video files: {len(missing)}")

    print("\nunique values per column:")
    print(df.drop(columns=["video_path"]).nunique().to_string())

    const = [c for c in df.columns if c != "video_path" and df[c].nunique() == 1]
    print(f"\nconstant columns: {const}")

    # Label range and clips per label value
    counts = df["target"].value_counts()
    print(
        f"\ntarget ({TARGET_COL[variable]}): {df['target'].nunique()} unique values, "
        f"min={df['target'].min():.4g}, max={df['target'].max():.4g}"
    )
    print(f"clips per value: min={counts.min()}, max={counts.max()}, "
          f"counts={sorted(counts.unique().tolist())}")

    # Ranges of the other (nuisance) fields
    print("\nnuisance field ranges:")
    for c in ["start_x", "start_y", "theta_degrees", "speed_mps", "acceleration_mps2"]:
        if c in df.columns:
            print(f"  {c}: min={df[c].min():.4g}, max={df[c].max():.4g}, "
                  f"unique={df[c].nunique()}")

    print("\nmotion values:", df["motion"].unique().tolist())
    print("\nhead:")
    print(df.drop(columns=["video_path"]).head(3).to_string())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("variables", nargs="*", default=VARIABLES)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for v in args.variables:
        inspect(v, load_dataset(v, args.data_root))