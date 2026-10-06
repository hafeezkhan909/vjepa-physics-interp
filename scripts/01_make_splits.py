import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.model_selection import train_test_split

from src.data import VARIABLES, load_dataset


def make_split(df, seed, test_size):
    # Stratify on the label so every class appears in both splits
    train_ids, test_ids = train_test_split(
        df["id"].values,
        test_size=test_size,
        stratify=df["target"].values,
        random_state=seed,
    )
    return sorted(int(i) for i in train_ids), sorted(int(i) for i in test_ids)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.cache_dir.mkdir(exist_ok=True)

    for v in VARIABLES:
        df = load_dataset(v)
        train_ids, test_ids = make_split(df, args.seed, args.test_size)

        # Sanity checks: no overlap, every clip assigned
        assert not set(train_ids) & set(test_ids)
        assert len(train_ids) + len(test_ids) == len(df)

        # Report per-class counts to confirm stratification
        by_id = df.set_index("id")["target"]
        tr_counts = by_id.loc[train_ids].value_counts()
        te_counts = by_id.loc[test_ids].value_counts()
        print(
            f"{v}: train={len(train_ids)}, test={len(test_ids)} | "
            f"per-class train {tr_counts.min()}-{tr_counts.max()}, "
            f"test {te_counts.min()}-{te_counts.max()} | "
            f"classes in test: {len(te_counts)}/{df['target'].nunique()}"
        )

        # Save clip ids for this variable
        with open(args.cache_dir / f"splits_{v}.json", "w") as f:
            json.dump(
                {"seed": args.seed, "test_size": args.test_size,
                 "train": train_ids, "test": test_ids},
                f,
            )