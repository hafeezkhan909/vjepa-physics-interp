import argparse
import sys
from pathlib import Path

import av
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoModel, AutoVideoProcessor

from src.data import VARIABLES, load_dataset

T_SLOTS = 8   # 16 frames / tubelet size 2
GRID = 16     # 256 / patch size 16


def read_video(path: str) -> torch.Tensor:
    # Decode mp4 -> uint8 [T, C, H, W], native frame count
    with av.open(path) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()


def preprocess(video: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    # Clips are already 256x256, so only normalize (HF processor fast path needs torch>=2.3)
    assert video.shape[-2:] == (256, 256), video.shape
    return (video.float() / 255.0 - mean) / std


@torch.inference_mode()
def extract(variable, model, processor, device, cache_dir, batch_size, limit=None):
    mean = torch.tensor(processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, dtype=torch.float32).view(1, 3, 1, 1)
    print("normalization mean/std:", processor.image_mean, processor.image_std)

    # Skip if already cached (unless doing a test run)
    out = cache_dir / f"feats_{variable}.npz"
    if out.exists() and limit is None:
        print(f"{out.name} exists, skipping")
        return
    df = load_dataset(variable)
    if limit:
        df = df.iloc[:limit]

    feats_mean, feats_last = [], []
    for i in tqdm(range(0, len(df), batch_size), desc=variable):
        paths = df["video_path"].iloc[i : i + batch_size]
        px = torch.stack([preprocess(read_video(p), mean, std) for p in paths]).to(device)
        if i == 0:
            print("pixel_values_videos:", tuple(px.shape))  # expect [B, 16, 3, 256, 256]

        # Hidden states: patch embedding + 24 blocks = 25
        hs = model.encoder(pixel_values_videos=px, output_hidden_states=True).hidden_states
        if i == 0:
            print("num hidden states:", len(hs), "| tokens:", hs[0].shape[1])  # expect 25, 2048
        B, n_tok, D = hs[0].shape
        assert n_tok == T_SLOTS * GRID * GRID, f"unexpected token count {n_tok}"

        # Pool each layer two ways
        mean_pool, last_pool = [], []
        for h in hs:
            h = h.float()
            mean_pool.append(h.mean(dim=1))  # all tokens
            # tokens are time-major: [T=8, H=16, W=16]; keep last time slot, mean over space
            last_pool.append(h.view(B, T_SLOTS, GRID, GRID, D)[:, -1].mean(dim=(1, 2)))
        feats_mean.append(torch.stack(mean_pool, dim=1).half().cpu().numpy())  # [B, 25, 1024]
        feats_last.append(torch.stack(last_pool, dim=1).half().cpu().numpy())  # [B, 25, 1024]

    feats_mean = np.concatenate(feats_mean)
    feats_last = np.concatenate(feats_last)

    # Test run: print shapes
    if limit:
        print("test run, shapes:", feats_mean.shape, feats_last.shape)
        return

    cache_dir.mkdir(exist_ok=True)
    np.savez(
        out,
        feats_mean=feats_mean,
        feats_last=feats_last,
        ids=df["id"].values,
        target=df["target"].values,
    )
    print(f"saved {out.name}: {feats_mean.shape}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("variables", nargs="*", default=VARIABLES)
    p.add_argument("--model-id", default="facebook/vjepa2-vitl-fpc64-256")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    p.add_argument("--limit", type=int, default=None)  # quick test on first N clips
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.model_id).to(device).eval()
    processor = AutoVideoProcessor.from_pretrained(args.model_id)
    for v in args.variables:
        extract(v, model, processor, device, args.cache_dir, args.batch_size, args.limit)