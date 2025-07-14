#!/usr/bin/env python3
"""
Run InferLetter (exp-3) in parallel.
Each worker handles **one seed** and iterates over all seq_ids.

Usage:
    python exp3_InferLetterAllModels.py \
        --base-config path/to/config.toml \
        --seq 0-25 \
        --seed 1-10 \
        --num-workers 10
"""

import argparse
import copy
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import multiprocessing as mp

import toml


# ───────────────────────── helpers ──────────────────────────
def repo_root() -> Path:
    """Return repository root (= two levels above this file)."""
    return Path(__file__).resolve().parents[1]


def resolve_path(p: str | os.PathLike) -> Path:
    """Make *p* absolute (relative paths are resolved from repo root)."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else repo_root() / p


def parse_range(spec: str) -> list[int]:
    """Convert '1,3,5-7' → [1,3,5,6,7]."""
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-"))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return out


# ────────────────────── worker function ─────────────────────
def run_for_seed(seed: int, seq_ids: list[int], base_cfg: dict, robot: bool):
    """
    Worker: run all required seq_ids for a single *seed*.
    Saves each run under   Data/exp3_InferLetter/Seed{seed}/Seq_{seq}/
    """
    for seq_id in seq_ids:
        cfg = copy.deepcopy(base_cfg)
        ocfg = cfg["onlineER"]

        save_dir = resolve_path(f"Data/exp3_InferLetter/Seed{seed}/Seq_{seq_id}")
        save_dir.mkdir(parents=True, exist_ok=True)
        ocfg["inferLetter_save_directory"] = str(save_dir)

        # create unique temp-TOML
        with tempfile.NamedTemporaryFile(
            suffix=f"_seq{seq_id}_seed{seed}.toml",
            delete=False,
            mode="w",
        ) as tf:
            toml.dump(cfg, tf)
            tmp_toml = tf.name

        # build command
        cmd = [
            sys.executable, "-m", "experiments.d3_InferLetter",
            "--config", tmp_toml,
            "--seq-id", str(seq_id),
            "--rand_seed", str(seed),
        ]
        if robot:
            cmd.append("--robot")

        print(f"[Seed {seed}] ▶ seq {seq_id} → {save_dir}")  # log
        try:
            subprocess.run(cmd, check=True)
        finally:
            os.remove(tmp_toml)  # always clean up


# ─────────────────────────── main ───────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--seq", default="0-25")
    ap.add_argument("--seed", default="1-10")
    ap.add_argument("--robot", action="store_true")
    ap.add_argument("--num-workers", type=int,
                    help="max parallel workers (default = #seeds)")
    args = ap.parse_args()

    seq_ids = parse_range(args.seq)
    seeds   = parse_range(args.seed)
    workers = args.num_workers or len(seeds)

    base_cfg = toml.load(args.base_config)

    # Each process handles one seed
    with mp.Pool(processes=workers) as pool:
        pool.starmap(
            run_for_seed,
            [(s, seq_ids, base_cfg, args.robot) for s in seeds]
        )


if __name__ == "__main__":
    main()
