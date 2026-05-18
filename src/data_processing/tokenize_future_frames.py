"""
One-off preprocessing script: tokenize future frames for the future-frame auxiliary decoder.

LeRobot dataset layout assumed:
  {root}/data/chunk-{N}/file-{M}.parquet  — per-frame metadata
  {root}/videos/{camera_key}/chunk-{N}/file-{M}.mp4  — video chunks

Runs in two phases:

  Phase 1 — Train VQ-VAE on frames from all cameras (learns robot-specific codebook).
             Saves checkpoint. Only needs to run ONCE.

  Phase 2 — Encode future frames to token IDs and save per-episode .npy files.
             For episode timestep t, encodes frame at min(t + n_action_steps, T_episode-1).
             Output: {root}/future_frame_tokens/{camera_key}/episode_{idx:06d}.npy
                     shape (T_episode, 256), dtype int16

Usage:
  # Full pipeline (train then tokenize all cameras):
  python src/data_processing/tokenize_future_frames.py \\
    --dataset-root ./output \\
    --cameras observation.images.front observation.images.gripper observation.images.right \\
    --n-action-steps 8

  # Tokenize only (reuse existing checkpoint):
  python src/data_processing/tokenize_future_frames.py \\
    --dataset-root ./output \\
    --cameras observation.images.front observation.images.gripper observation.images.right \\
    --n-action-steps 8 \\
    --tokenizer-ckpt ./output/frame_tokenizer/vqvae.pt \\
    --skip-training

  # Train VQ-VAE only:
  python src/data_processing/tokenize_future_frames.py \\
    --dataset-root ./output \\
    --cameras observation.images.front \\
    --n-action-steps 8 \\
    --train-only
"""

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.transformer_flow_matching.frame_tokenizer import RobotFrameVQVAE


# ---------------------------------------------------------------------------
# LeRobot frame index — maps global index ↔ episode/frame/video position
# ---------------------------------------------------------------------------

def build_frame_index(dataset_root: Path) -> "pd.DataFrame":
    """
    Read all data parquet files and return a combined, index-sorted DataFrame.

    Columns returned:
      index          — global frame index (unique, 0-based across all episodes)
      episode_index  — which episode this frame belongs to
      frame_index    — within-episode counter (resets to 0 each episode)
      chunk_id       — which chunk file the frame lives in
      local_pos      — frame position within that chunk's video file

    The frame at local_pos P in chunk C's video == the P-th row (sorted by
    global index) of that chunk's parquet. This matches how LeRobot writes mp4s.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required: pip install pandas pyarrow")

    chunk_dfs = []
    chunk_dirs = sorted((dataset_root / "data").glob("chunk-*"))
    if not chunk_dirs:
        raise RuntimeError(f"No data/chunk-* directories found in {dataset_root}")

    for chunk_dir in chunk_dirs:
        chunk_id = int(chunk_dir.name.split("-")[1])
        for pq_file in sorted(chunk_dir.glob("file-*.parquet")):
            import pandas as pd
            df = pd.read_parquet(pq_file, columns=["index", "episode_index", "frame_index"])
            df = df.sort_values("index").reset_index(drop=True)
            df["chunk_id"]  = chunk_id
            df["local_pos"] = df.index   # row position = video frame position
            chunk_dfs.append(df)

    if not chunk_dfs:
        raise RuntimeError(f"No parquet files found under {dataset_root / 'data'}")

    import pandas as pd
    frame_df = pd.concat(chunk_dfs, ignore_index=True).sort_values("index").reset_index(drop=True)
    print(f"[Frame index] {len(frame_df):,} frames | "
          f"{frame_df['episode_index'].nunique()} episodes | "
          f"{frame_df['chunk_id'].nunique()} chunk(s)")
    return frame_df


def find_chunk_videos(dataset_root: Path, camera_key: str) -> dict:
    """
    Return {chunk_id: Path} for each chunk video file for a given camera.
    Layout: videos/{camera_key}/chunk-{N}/file-{M}.mp4
    """
    pattern = str(dataset_root / "videos" / camera_key / "chunk-*" / "file-*.mp4")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise RuntimeError(
            f"No video files found for camera '{camera_key}'.\n"
            f"Expected: {dataset_root}/videos/{camera_key}/chunk-*/file-*.mp4"
        )
    result = {}
    for p in paths:
        p = Path(p)
        chunk_id = int(p.parent.name.split("-")[1])
        result[chunk_id] = p
    return result


def load_chunk_frames(video_path: Path, target_size: int = 128) -> list[np.ndarray]:
    """
    Load ALL frames from one chunk .mp4 into a list of resized (H, W, 3) uint8 arrays.
    Frames are returned in video order, which matches the parquet row order for that chunk.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[0] != target_size or frame.shape[1] != target_size:
            frame = cv2.resize(frame, (target_size, target_size),
                               interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    return frames


def load_all_frames_for_camera(
    dataset_root: Path,
    camera_key: str,
    frame_df: "pd.DataFrame",
    target_size: int = 128,
) -> np.ndarray:
    """
    Load every frame for one camera into a flat numpy array indexed by global `index`.

    Returns: np.ndarray of shape (N_total_frames, H, W, 3) uint8
             where arr[global_index] = that frame's pixels.

    Memory: ~1.4 GB for 28k frames at 128×128×3.  Acceptable for a one-off script.
    """
    chunk_video_map = find_chunk_videos(dataset_root, camera_key)
    n_total = int(frame_df["index"].max()) + 1
    frame_store = np.zeros((n_total, target_size, target_size, 3), dtype=np.uint8)

    for chunk_id, video_path in tqdm(chunk_video_map.items(),
                                      desc=f"  Loading {camera_key}", ncols=80):
        chunk_rows = frame_df[frame_df["chunk_id"] == chunk_id].sort_values("index")
        global_indices = chunk_rows["index"].tolist()

        chunk_frames = load_chunk_frames(video_path, target_size)
        if len(chunk_frames) != len(global_indices):
            print(f"  Warning: chunk {chunk_id} has {len(chunk_frames)} video frames "
                  f"but {len(global_indices)} parquet rows. Using min.")
        for local_pos, global_idx in enumerate(global_indices):
            if local_pos < len(chunk_frames):
                frame_store[global_idx] = chunk_frames[local_pos]

    return frame_store   # (N, 128, 128, 3) uint8


# ---------------------------------------------------------------------------
# Phase 1 — VQ-VAE training dataset (all frames, all cameras)
# ---------------------------------------------------------------------------

class AllFramesDataset(Dataset):
    """
    Flat dataset of frames from all chunk videos across all cameras.
    Used for VQ-VAE training. Every 3rd frame is taken to reduce near-duplicates.
    """
    def __init__(self, dataset_root: Path, cameras: list[str], target_size: int = 128):
        self.target_size = target_size
        self.frames: list[np.ndarray] = []

        for cam in cameras:
            try:
                chunk_map = find_chunk_videos(dataset_root, cam)
            except RuntimeError as e:
                print(f"  Skipping camera {cam}: {e}")
                continue
            for chunk_id, vp in sorted(chunk_map.items()):
                all_f = load_chunk_frames(vp, target_size)
                # Subsample every 3rd frame to avoid near-duplicates at 30fps
                self.frames.extend(all_f[::3])

        if not self.frames:
            raise RuntimeError("No frames loaded — check dataset_root and camera names.")
        print(f"[VQ-VAE train] {len(self.frames):,} frames from {len(cameras)} camera(s)")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.frames[idx]).permute(2, 0, 1).float() / 255.0
        if torch.rand(1).item() < 0.5:
            img = img.flip(-1)   # random horizontal flip (only safe augmentation)
        return img


# ---------------------------------------------------------------------------
# Phase 1 — Training loop
# ---------------------------------------------------------------------------

def train_vqvae(
    dataset_root: Path,
    cameras: list[str],
    save_path: Path,
    codebook_size: int = 1024,
    latent_dim: int = 256,
    batch_size: int = 64,
    epochs: int = 30,
    lr: float = 1e-4,
    device: str = "cuda",
) -> RobotFrameVQVAE:
    dataset = AllFramesDataset(dataset_root, cameras)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=4, pin_memory=True, drop_last=True)

    model     = RobotFrameVQVAE(codebook_size=codebook_size, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss, patience_counter, patience = float("inf"), 0, 5
    print(f"\n[VQ-VAE] Training — codebook={codebook_size}, latent_dim={latent_dim}, "
          f"{epochs} epochs max")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch:02d}/{epochs}", ncols=80, leave=False):
            batch = batch.to(device)
            _, loss, _ = model(batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        avg = epoch_loss / len(loader)
        print(f"  Epoch {epoch:02d} | loss={avg:.4f}")

        if avg < best_loss - 1e-4:
            best_loss, patience_counter = avg, 0
            model.save(save_path)
        else:
            patience_counter += 1

        if epoch >= 10 and patience_counter >= patience:
            print(f"  [VQ-VAE] Early stop at epoch {epoch}")
            break

    print(f"[VQ-VAE] Done. Best loss={best_loss:.4f}  Checkpoint → {save_path}")
    return RobotFrameVQVAE.load(save_path, device=device)


# ---------------------------------------------------------------------------
# Phase 2 — Tokenize future frames for all cameras
# ---------------------------------------------------------------------------

def tokenize_camera(
    dataset_root: Path,
    camera_key: str,
    frame_df: "pd.DataFrame",
    model: RobotFrameVQVAE,
    n_action_steps: int,
    device: str,
    overwrite: bool = False,
    encode_batch_size: int = 64,
) -> None:
    """
    For one camera, load all frames then for each episode write a .npy token file.
    Token shape: (T_episode, 256) int16
    """
    out_dir = dataset_root / "future_frame_tokens" / camera_key
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if all episodes are already done
    episode_ids = sorted(frame_df["episode_index"].unique())
    pending = [ep for ep in episode_ids
               if overwrite or not (out_dir / f"episode_{ep:06d}.npy").exists()]
    if not pending:
        print(f"  {camera_key}: all {len(episode_ids)} episodes already tokenized, skipping")
        return

    print(f"\n  Camera: {camera_key}")
    print(f"  Episodes to process: {len(pending)} / {len(episode_ids)}")

    # Load all frames for this camera into memory (indexed by global `index`)
    print("  Loading frames into memory...")
    frame_store = load_all_frames_for_camera(dataset_root, camera_key, frame_df)

    model.eval()
    for ep_idx in tqdm(pending, desc=f"  Tokenizing {camera_key.split('.')[-1]}", ncols=80):
        ep_rows = frame_df[frame_df["episode_index"] == ep_idx].sort_values("frame_index")
        global_indices = ep_rows["index"].tolist()   # ordered by frame_index
        T = len(global_indices)

        all_tokens = np.zeros((T, RobotFrameVQVAE.N_TOKENS), dtype=np.int16)

        for start in range(0, T, encode_batch_size):
            end = min(start + encode_batch_size, T)
            future_indices = [global_indices[min(t + n_action_steps, T - 1)]
                              for t in range(start, end)]

            batch = np.stack([frame_store[i] for i in future_indices])    # (B, H, W, 3)
            tensor = (torch.from_numpy(batch).permute(0, 3, 1, 2)
                      .float().div(255.0).to(device))                      # (B, 3, H, W)

            token_ids = model.encode(tensor).cpu().numpy().astype(np.int16)
            all_tokens[start:end] = token_ids

        np.save(out_dir / f"episode_{ep_idx:06d}.npy", all_tokens)

    print(f"  Done → {out_dir}")


def tokenize_all_cameras(
    dataset_root: Path,
    cameras: list[str],
    frame_df: "pd.DataFrame",
    model: RobotFrameVQVAE,
    n_action_steps: int,
    device: str,
    overwrite: bool = False,
) -> None:
    for cam in cameras:
        try:
            tokenize_camera(dataset_root, cam, frame_df, model,
                            n_action_steps, device, overwrite)
        except RuntimeError as e:
            print(f"  ERROR for {cam}: {e}")


# ---------------------------------------------------------------------------
# Verification — reconstruction sanity check
# ---------------------------------------------------------------------------

@torch.no_grad()
def verify_reconstruction(
    model: RobotFrameVQVAE,
    dataset_root: Path,
    cameras: list[str],
    save_dir: Path,
    n_samples: int = 4,
    device: str = "cuda",
) -> None:
    try:
        import torchvision.utils as vutils
    except ImportError:
        print("[Verify] torchvision not installed, skipping reconstruction check")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    collected = []
    for cam in cameras:
        if len(collected) >= n_samples:
            break
        try:
            chunk_map = find_chunk_videos(dataset_root, cam)
        except RuntimeError:
            continue
        vp = next(iter(chunk_map.values()))
        frames = load_chunk_frames(vp, RobotFrameVQVAE.INPUT_SIZE)
        # Pick evenly spaced frames from this camera
        picks = np.linspace(0, len(frames) - 1, min(n_samples, len(frames)), dtype=int)
        for i in picks:
            collected.append(frames[i])
            if len(collected) >= n_samples:
                break

    if not collected:
        print("[Verify] No frames found for reconstruction check")
        return

    imgs = (torch.from_numpy(np.stack(collected))
            .permute(0, 3, 1, 2).float().div(255.0).to(device))
    recons, _, _ = model(imgs)

    # Interleave: [orig, recon, orig, recon, ...]
    pairs = []
    for orig, recon in zip(imgs.cpu(), recons.cpu()):
        pairs.extend([orig, recon])

    grid = vutils.make_grid(torch.stack(pairs), nrow=4, padding=2)
    out  = save_dir / "vqvae_reconstruction_check.png"
    vutils.save_image(grid, out)
    print(f"[Verify] Grid saved → {out}")
    print("         Odd columns = original | Even columns = reconstructed")
    print("         Blurry-but-recognisable scene structure = codebook is working.")
    print("         Grey blobs = codebook collapsed, reduce LR or train longer.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root",   required=True,       type=Path)
    parser.add_argument("--cameras",        required=True, nargs="+",
                        help="Camera keys, e.g. observation.images.front observation.images.gripper")
    parser.add_argument("--n-action-steps", default=8,           type=int)
    parser.add_argument("--tokenizer-ckpt", default=None,        type=Path,
                        help="Path to save/load VQ-VAE checkpoint")
    parser.add_argument("--skip-training",  action="store_true",
                        help="Skip Phase 1 (VQ-VAE training), load existing checkpoint")
    parser.add_argument("--train-only",     action="store_true",
                        help="Run Phase 1 only (no tokenization)")
    parser.add_argument("--overwrite",      action="store_true",
                        help="Re-encode episodes even if output already exists")
    parser.add_argument("--codebook-size",  default=1024,        type=int)
    parser.add_argument("--latent-dim",     default=256,         type=int)
    parser.add_argument("--batch-size",     default=64,          type=int)
    parser.add_argument("--epochs",         default=30,          type=int)
    parser.add_argument("--lr",             default=1e-4,        type=float)
    parser.add_argument("--no-verify",      action="store_true")
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"
    print(f"[Device] {device}")

    if args.tokenizer_ckpt is None:
        args.tokenizer_ckpt = args.dataset_root / "frame_tokenizer" / "vqvae.pt"

    # ---- Phase 1: Train or load VQ-VAE ----
    if args.skip_training:
        if not args.tokenizer_ckpt.exists():
            print(f"ERROR: checkpoint not found: {args.tokenizer_ckpt}")
            sys.exit(1)
        model = RobotFrameVQVAE.load(args.tokenizer_ckpt, device=device)
    else:
        model = train_vqvae(
            dataset_root=args.dataset_root,
            cameras=args.cameras,
            save_path=args.tokenizer_ckpt,
            codebook_size=args.codebook_size,
            latent_dim=args.latent_dim,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
        )

    if not args.no_verify:
        verify_reconstruction(model, args.dataset_root, args.cameras,
                              save_dir=args.dataset_root / "frame_tokenizer",
                              device=device)

    if args.train_only:
        print("[Phase 2] Skipped (--train-only)")
        return

    # ---- Phase 2: Build frame index and tokenize ----
    frame_df = build_frame_index(args.dataset_root)
    tokenize_all_cameras(
        dataset_root=args.dataset_root,
        cameras=args.cameras,
        frame_df=frame_df,
        model=model,
        n_action_steps=args.n_action_steps,
        device=device,
        overwrite=args.overwrite,
    )

    # Summary
    print("\n=== Done ===")
    for cam in args.cameras:
        out_dir = args.dataset_root / "future_frame_tokens" / cam
        files   = list(out_dir.glob("episode_*.npy")) if out_dir.exists() else []
        print(f"  {cam}: {len(files)} token files")
        if files:
            s = np.load(files[0])
            print(f"    shape={s.shape}  dtype={s.dtype}  "
                  f"token range=[{s.min()},{s.max()}]/{args.codebook_size}")
    print(f"\nToken files: {{dataset_root}}/future_frame_tokens/{{camera}}/episode_XXXXXX.npy")


if __name__ == "__main__":
    main()
