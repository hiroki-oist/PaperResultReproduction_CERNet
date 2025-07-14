"""
pcrnn_viz.py
====================
Utility functions to visualise PC‑RNN training runs saved by *PC_RNN_HC_A*.

The module can be **imported as a library** or executed as a
command‑line program:

```bash
# programmatic use
from pcrnn_viz import generate_videos_from_config

generate_videos_from_config("configs/reachy_experiment.toml", fps=12)

# CLI use (equivalent)
python -m pcrnn_viz configs/reachy_experiment.toml --fps 12
```

It expects the following directory structure (same as the training script):

```
$PVRNN_SAVE_DIR/<training.save_directory>/seed*/iteration_*/snapshot.pkl
```

Each *seed* folder receives a video file **snapshot_video_allResult.mp4**
containing one frame per iteration.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import pickle as pk
from functools import partial
from pathlib import Path
from typing import List, Tuple

import imageio.v2 as imageio  # ensure legacy API
import matplotlib.pyplot as plt
import numpy as np
import toml

__all__ = [
    "generate_videos_from_config",
]

# ---------------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------------

def _list_iteration_dirs(base_dir: Path) -> List[Path]:
    """Return iteration_* directories ordered by iteration number."""
    return sorted(
        [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("iteration_")],
        key=lambda p: int(p.name.split("_")[-1]),
    )


def _extract_iter_num(path: Path | str) -> int:
    """Return the integer iteration number encoded in the filename or directory."""
    name = Path(path).name
    return int(name.split("_")[-1].split(".")[0])


def _compute_error_metrics(seed_dir: Path, *, use_free: bool = False) -> Tuple[List[int], List[float]]:
    """Compute MSE (sum over dims) per iteration.

    Parameters
    ----------
    seed_dir : Path
        Path to a *seed* directory containing iteration sub‑directories.
    use_free : bool, default False
        If *True*, use ``error_free`` from snapshots (α=0, open loop).
        Otherwise use training error ``error``.
    """
    xs, ys = [], []
    for it_dir in _list_iteration_dirs(seed_dir):
        snap_file = it_dir / "snapshot.pkl"
        if not snap_file.exists():
            continue
        with snap_file.open("rb") as f:
            snap = pk.load(f)
        key = "error_free" if use_free else "error"
        err = snap.get(key)
        if err is None:
            continue
        err = np.asarray(err)
        mse = np.mean(np.sum(err ** 2, axis=-1))
        xs.append(_extract_iter_num(it_dir))
        ys.append(float(mse))
    return xs, ys


def _plot_snapshot(
    snapshot: dict,
    iteration: int,
    out_png: Path,
    *,
    target_xy: np.ndarray | None,
    err_curve: Tuple[List[int], List[float]] | None,
) -> None:
    """Render one snapshot to *out_png*."""
    # Weights ---------------------------------------------------------------
    w_r = snapshot["w_r"]
    w_hh = snapshot["w_hh"]
    n_weight = 2 + len(w_r) + len(w_hh)  # w_o, w_c, w_r*, w_hh*
    n_output = 6  # x_pred, x_pred_free, error, error_free, h_prior0, c
    n_tot = n_weight + n_output
    n_cols = 4
    n_rows = math.ceil(n_tot / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    a = 0  # current axis index

    def _imshow(mat, title):
        nonlocal a
        im = axes[a].imshow(np.abs(mat), aspect="auto", cmap="viridis")
        axes[a].set_title(title)
        plt.colorbar(im, ax=axes[a])
        a += 1

    _imshow(snapshot["w_o"], "w_o (|·|)")
    _imshow(snapshot["w_c"], "w_c (|·|)")
    for i, wr in enumerate(w_r):
        _imshow(wr, f"w_r[{i}] (|·|)")
    for i, whh in enumerate(w_hh):
        _imshow(whh, f"w_hh[{i}] (|·|)")

    # Trajectories ----------------------------------------------------------
    def _plot_traj(arr, label, color):
        nonlocal a
        if arr is None:
            axes[a].text(0.5, 0.5, f"{label} N/A", ha="center", va="center")
        else:
            arr = np.asarray(arr)[:, 0, :]  # (T, 2)
            axes[a].plot(arr[:, 0], -arr[:, 1], label=label, c=color)
            if target_xy is not None:
                axes[a].plot(target_xy[:, 0], -target_xy[:, 1], "--", c="k", label="target")
            axes[a].set_aspect("equal")
            axes[a].legend(fontsize="small")
        axes[a].set_title(label)
        a += 1

    _plot_traj(snapshot.get("x_pred"), "x_pred", "tab:blue")
    _plot_traj(snapshot.get("x_pred_free"), "x_pred_free", "tab:orange")

    # Error curves ----------------------------------------------------------
    for key in ("error", "error_free"):
        if key not in snapshot:
            continue
        err = snapshot[key]
        if err is None:
            axes[a].text(0.5, 0.5, f"{key} N/A", ha="center", va="center")
        else:
            curve = np.sum(np.asarray(err) ** 2, axis=-1).mean(axis=1)
            axes[a].plot(curve)
            axes[a].set_yscale("log")
        axes[a].set_title(key)
        a += 1

    # h_prior[0][:,0,0] ----------------------------------------------------
    hp = snapshot.get("h_prior")
    if hp is not None:
        axes[a].plot(np.asarray(hp[0])[:, 0, 0])
    else:
        axes[a].text(0.5, 0.5, "N/A", ha="center", va="center")
    axes[a].set_title("h_prior[0] unit0"); a += 1

    # cause series ---------------------------------------------------------
    c = snapshot.get("c")
    if c is not None:
        c = np.asarray(c)[:, 0, :]
        for d in range(c.shape[1]):
            axes[a].plot(c[:, d], label=f"C{d}")
        axes[a].legend(fontsize="x-small")
    else:
        axes[a].text(0.5, 0.5, "N/A", ha="center", va="center")
    axes[a].set_title("cause"); a += 1

    # global error metric ---------------------------------------------------
    if err_curve is not None:
        xs, ys = err_curve
        axes[a].plot(xs, ys)
        axes[a].set_yscale("log")
        axes[a].set_title("MSE vs iter")
    else:
        axes[a].axis("off")
    a += 1

    # hide leftover axes ----------------------------------------------------
    for i in range(a, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"Snapshot @ iter {iteration}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png)
    plt.close(fig)


def _process_one_iteration(
    it_dir: Path,
    *,
    target_xy: np.ndarray | None,
    err_curve: Tuple[List[int], List[float]] | None,
) -> str | None:
    snap_file = it_dir / "snapshot.pkl"
    if not snap_file.exists():
        return None
    with snap_file.open("rb") as f:
        snapshot = pk.load(f)
    iter_num = _extract_iter_num(it_dir)
    out_png = it_dir / f"snapshot_{iter_num}.png"
    _plot_snapshot(snapshot, iter_num, out_png, target_xy=target_xy, err_curve=err_curve)
    return str(out_png)


def _make_video(images: List[str], out_path: Path, fps: int):
    if not images:
        print("[warn] no frames -> skip video")
        return
    with imageio.get_writer(out_path, fps=fps) as writer:
        for img in sorted(images, key=_extract_iter_num):
            writer.append_data(imageio.imread(img))
    try:
        rel = out_path.relative_to(Path.cwd())
    except ValueError:
        rel = out_path          # cwd 直下でなければ絶対パス
    print(f"[video] {rel}")

# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def generate_videos_from_config(toml_path: str | os.PathLike, *, fps: int = 10) -> None:
    """Main entry point usable from another script.

    Parameters
    ----------
    toml_path : str or Path
        Path to the training configuration TOML file.
    fps : int, default 10
        Frames per second of the resulting mp4.
    """
    toml_path = Path(toml_path)
    cfg = toml.load(toml_path)

    # fetch paths ----------------------------------------------------------
    save_dir = cfg["training"]["save_directory"]
    dataset_path = Path(cfg["dataset"]["dataset_path"])

    root = os.environ.get("PVRNN_SAVE_DIR")
    if root is None:
        raise EnvironmentError("env PVRNN_SAVE_DIR is not set")
    exp_root = Path(root) / save_dir
    if not exp_root.exists():
        raise FileNotFoundError(exp_root)

    # discover seed directories -------------------------------------------
    seed_dirs = sorted([p for p in exp_root.iterdir() if p.is_dir() and p.name.startswith("seed")])
    if not seed_dirs:
        raise RuntimeError(f"no seed* dirs under {exp_root}")

    # normalise target trajectory once, shared by all seeds ---------------
    if dataset_path.is_file():
        data = np.load(dataset_path)
        data_min, data_max = data.min(), data.max()
        data_norm = -0.9 + (data - data_min) * (1.8 / (data_max - data_min))
        target_xy = data_norm[0]  # first sequence, (T,2)
    else:
        target_xy = None

    # iterate seeds --------------------------------------------------------
    for seed_dir in seed_dirs:
        print(f"\n[seed] {seed_dir.relative_to(exp_root)}")

        # compute error curve once
        xs, ys = _compute_error_metrics(seed_dir, use_free=False)
        np.save(seed_dir / "error_metrics.npy", np.vstack([xs, ys]))
        err_curve: Tuple[List[int], List[float]] | None = (xs, ys)

        # parallel plotting over iterations
        it_dirs = _list_iteration_dirs(seed_dir)
        func = partial(_process_one_iteration, target_xy=target_xy, err_curve=err_curve)
        with mp.Pool() as pool:
            pngs = [p for p in pool.map(func, it_dirs) if p is not None]

        # assemble video
        _make_video(pngs, seed_dir / "snapshot_video_allResult.mp4", fps=fps)

# ---------------------------------------------------------------------------
# CLI glue
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(description="Generate training‑snapshot videos from a PC‑RNN config.")
    parser.add_argument("toml_config", help="Path to TOML config used for training")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second of output video")
    args = parser.parse_args()
    generate_videos_from_config(args.toml_config, fps=args.fps)


if __name__ == "__main__":
    _cli()
