#!/usr/bin/env python3
"""
Batch runner for `experiments.d2_PerturbationWhileGeneration`.

It launches the target module N times using the same TOML config and
relies on the default values for `--perturbation_timestep`,
`--class_id`, and `--robot` unless the user overrides them via CLI.
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------
def run_once(script: str, config_path: Path, robot: bool) -> int:
    """Spawn a single subprocess and return its exit code."""
    cmd = [
        sys.executable,
        "-m",
        script,
        "--pcrnn_config",
        str(config_path),
    ]
    if robot:
        cmd.append("--robot")
    print("Executing:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


# ----------------------------------------------------------------------
# Main CLI
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run d2_PerturbationWhileGeneration multiple times."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of repetitions (default: 20)",
    )
    parser.add_argument(
        "--config",
        default="./toml_configs/multi_large.toml",
        help="Path passed to --pcrnn_config",
    )
    parser.add_argument(
        "--script",
        default="experiments.d2_PerturbationWhileGeneration",
        help="Target module executed with -m",
    )
    parser.add_argument(
        "--robot",
        action="store_true",
        help="Forward the --robot flag to the child process",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        sys.exit(f"Config not found: {cfg_path}")

    for idx in range(1, args.runs + 1):
        print(f"\n=== Run {idx}/{args.runs} ===")
        rc = run_once(args.script, cfg_path, args.robot)
        if rc != 0:
            sys.exit(f"Aborted: run {idx} returned exit code {rc}")


if __name__ == "__main__":
    main()
