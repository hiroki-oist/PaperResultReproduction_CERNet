#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import toml  # pip install toml (or tomli+tomli-w for Python≥3.11)

import numpy as np
import matplotlib
matplotlib.use("Agg")         
import matplotlib.pyplot as plt

from utils.x1_path_utils import resolve_path

TRAIN_BASELINE_PATH = Path(
    resolve_path("Data/train_data_26_100_3.npy")
)
try:
    TRAIN_BASELINE = np.load(TRAIN_BASELINE_PATH)
except Exception as exc:
    print(f"[WARN] baseline load failed: {exc}")
    TRAIN_BASELINE = None

# ──────────────────────────────────────────────────────────────────────────────
# Constants (edit if paths or epochs change)
# ──────────────────────────────────────────────────────────────────────────────

MIN_LOSS_EPOCHS = {
    "MultiLarge": {
        1: 9801, 2: 9745, 3: 9999, 4: 8773, 5: 9492,
        6: 8543, 7: 9766, 8: 9980, 9: 9589, 10: 9866,
    },
    "MultiMini": {
        1: 9812, 2: 9896, 3: 9996, 4: 6297, 5: 7655,
        6: 6160, 7: 7647, 8: 7460, 9: 4586, 10: 7823,
    },
    "MultiStandard": {
        1: 9941, 2: 9428, 3: 9950, 4: 7812, 5: 7754,
        6: 9996, 7: 6385, 8: 4671, 9: 9998, 10: 9990,
    },
    "SingleLarge": {
        1: 9992, 2: 9996, 3: 9970, 4: 9996, 5: 9991,
        6: 2369, 7: 9998, 8: 9859, 9: 9997, 10: 9991,
    },
    "SingleMini": {
        1: 9716, 2: 9914, 3: 4929, 4: 7420, 5: 3658,
        6: 7047, 7: 9957, 8: 7426, 9: 2401, 10: 2301,
    },
    "SingleStandard": {
        1: 9999, 2: 9947, 3: 4902, 4: 4734, 5: 9997,
        6: 9976, 7: 9978, 8: 5078, 9: 7185, 10: 7538,
    },
}

# Base‑config stem → canonical model name
MODEL_NAME_MAP = {
    "single_mini": "SingleMini",
    "single_standard": "SingleStandard",
    "single_large": "SingleLarge",
    "multi_mini": "MultiMini",
    "multi_standard": "MultiStandard",
    "multi_large": "MultiLarge",
}

SEEDS = range(1,11)
CLASS_IDS = range(26) 
MODEL_CONFIGS = list(MODEL_NAME_MAP.keys())


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────

def _nearest_hundred(epoch: int) -> int:
    return (epoch // 100) * 100

def _plot_baseline_vs_pred(save_dir: Path, class_id: int):
    """baseline (train_data) と今回の予測を線で比較して PNG 保存."""
    if TRAIN_BASELINE is None:
        return

    base_xy = TRAIN_BASELINE[class_id, :, :2]          # shape: (N, 2)
    pred_f  = save_dir / f"pred_class_id_{class_id}.npy"
    if not pred_f.exists():
        print(f"[WARN] prediction file not found: {pred_f}")
        return
    pred_xy = np.load(pred_f)[:, :2]                   # (M, 2)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(-base_xy[:, 1], base_xy[:, 0],
            label="Baseline", linewidth=1.5)
    ax.plot(-pred_xy[:, 1], pred_xy[:, 0],
            label="Prediction", linewidth=1.5)

    ax.set_xlabel("-Y")
    ax.set_ylabel("X")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()
    out_png = save_dir / f"baseline_vs_pred_cid{class_id:02d}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"→ saved plot {out_png}")

def _snapshot_iter(model_name: str, seed: int) -> Optional[int]:
    """Return iteration number (nearest hundred before MinLoss) or None."""
    try:
        return _nearest_hundred(MIN_LOSS_EPOCHS[model_name][seed])
    except KeyError:
        return None


def _patch_model_path(path: str, seed: int, iteration: Optional[int]) -> str:
    path = re.sub(r"/seed\d+/", f"/seed{seed}/", path)
    if iteration is not None:
        path = re.sub(r"/iteration_\d+/", f"/iteration_{iteration}/", path)
    return path


def _prepare_config(base_cfg_path: Path, seed: int, class_id: int, tmp_dir: Path) -> tuple[Path, Path]:
    """Return (derived_toml_path, save_dir) for this run."""
    cfg = toml.load(base_cfg_path)
    stem        = base_cfg_path.stem  # e.g. "single_mini"
    model_name  = MODEL_NAME_MAP[stem]
    iteration   = _snapshot_iter(model_name, seed)

    # Patch fields ------------------------------------------------------
    mp_orig: str = cfg["onlineER"]["model_path"]
    cfg["onlineER"]["model_path"] = _patch_model_path(mp_orig, seed, iteration)

    cfg["onlineER"]["class_id"] = class_id

    save_dir = resolve_path(f"Data/exp1_SequenceGeneration/{model_name}/Seed{seed}")
    cfg["onlineER"]["SequenceGeneration_save_directory"] = str(save_dir)

    # Write out temp config
    out_cfg = tmp_dir / f"{stem}_seed{seed}_cid{class_id}.toml"
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    with out_cfg.open("w") as f:
        toml.dump(cfg, f)

    return out_cfg, save_dir


def _run_experiment(cfg_path: Path, robot: bool = False) -> int:
    cmd = [
        "python", "-m", "experiments.d1_SequenceGeneration",
        "--pcrnn_config", str(cfg_path)
    ]
    if robot:
        cmd.append("--robot")
    print("⟲", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def _rename_outputs(save_dir: Path, class_id: int):
    """Rename timestamped npy files to deterministic names."""
    # glob matches e.g. pred_xyz_20250522_123456.npy
    for kind in ("pred", "obs"):
        pattern = save_dir / f"{kind}_xyz_class{class_id:02d}_*.npy"
        files   = glob.glob(str(pattern))
        if not files:
            print(f"[WARN] {pattern} NOT FOUND")
            continue
        latest = max(files, key=os.path.getmtime)  # pick newest if several
        dst    = save_dir / f"{kind}_class_id_{class_id}.npy"
        try:
            shutil.move(latest, dst)
            print(f"→ renamed {latest} → {dst}")
        except Exception as exc:
            print(f"[WARN] rename failed: {exc}")


def _worker(args):
    base_cfg, seed, class_id, tmp_dir, robot = args
    try:
        cfg_path, save_dir = _prepare_config(base_cfg, seed, class_id, tmp_dir)
        exit_code = _run_experiment(cfg_path, robot=robot)
        if exit_code == 0:
            _rename_outputs(save_dir, class_id)
            _plot_baseline_vs_pred(save_dir, class_id)   # ★ここを追加★
        return cfg_path, exit_code
    except Exception as exc:
        return base_cfg, exc

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Launch Reachy sweep")
    p.add_argument("--config-dir", type=Path, default="./toml_configs")
    p.add_argument("--tmp-dir", type=Path, default="./tmp_autogen_configs")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--robot", action="store_true", help="Run in robot mode")
    args = p.parse_args()

    cfg_paths = [args.config_dir / f"{stem}.toml" for stem in MODEL_CONFIGS]
    tasks = [
        (cfg, seed, cid, args.tmp_dir, args.robot)
        for cfg in cfg_paths
        for seed in SEEDS
        for cid in CLASS_IDS
    ]

    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(_worker, tasks)

    # Summary -----------------------------------------------------------
    ok, fail = 0, 0
    for cfg_path, status in results:
        label = cfg_path.name if isinstance(cfg_path, Path) else str(cfg_path)
        if status == 0:
            ok += 1
        else:
            fail += 1
            print(f"FAIL: {label} ({status})")
    print(f"\nCompleted: {ok} OK, {fail} failed out of {len(results)} runs.")


if __name__ == "__main__":
    main()
