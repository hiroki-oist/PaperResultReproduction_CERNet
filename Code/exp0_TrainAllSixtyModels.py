import os
import subprocess
import multiprocessing as mp
from tqdm import tqdm
import re
import textwrap

from dotenv import load_dotenv
load_dotenv()

from utils.u4_animation import generate_videos_from_config

CONFIG_FILES = [
    "toml_configs/multi_mini.toml",
    "toml_configs/multi_large.toml",
    "toml_configs/multi_standard.toml",
    "toml_configs/single_mini.toml",
    "toml_configs/single_large.toml",
    "toml_configs/single_standard.toml"
]

SCRIPT_MODULE = "experiments.c1_TrainModel"

def run_training(config_path, seed):
    config_name = os.path.basename(config_path)
    print(f"[RUNNING] {config_name} | seed={seed}")

    cmd = [
        "python", "-u", "-m", SCRIPT_MODULE,
        "--config", config_path,
        "--seed", str(seed)
    ]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True)
        log = proc.stdout + proc.stderr

        m = re.search(r"Lowest Loss: ([\d\.]+) at Epoch (\d+)", log)
        success = m is not None and proc.returncode == 0
        return {
            "config": os.path.basename(config_path),
            "seed":   seed,
            "min_loss": float(m.group(1)) if success else None,
            "min_epoch": int(m.group(2))  if success else None,
            "returncode": proc.returncode,
            "stderr_tail": textwrap.shorten(proc.stderr, 300) 
        }

    except Exception as e:
        print(f"[ERROR] Failed: {config_name} | seed={seed}")
        return {
            "config": config_name,
            "seed": seed,
            "min_loss": None,
            "min_epoch": None,
            "error": True
        }

def main():
    num_processes = min(mp.cpu_count(), 10)
    tasks = [(cfg, seed) for cfg in CONFIG_FILES for seed in range(1, 11)]

    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.starmap(run_training, tasks), total=len(tasks)))
        print(results)
        
    for cfg in CONFIG_FILES:
        try:
            print(f"\n[VISUALISE] {cfg}")
            generate_videos_from_config(cfg, fps=12)   
        except Exception as e:
            print(f"[VISUALISE][ERROR] {cfg}: {e}")

    print("\n===== Summary of Minimum Losses =====")
    results_sorted = sorted(results, key=lambda r: (r["config"], r["seed"]))

    current_config = None
    for r in results_sorted:
        if r["config"] != current_config:
            current_config = r["config"]
            print(f"\nConfig: {current_config}")
        print(f"  Seed {r['seed']:2d}: MinLoss = {r['min_loss']:.6f} at Epoch {r['min_epoch']}"
              if r.get("min_loss") is not None
              else f"  Seed {r['seed']:2d}: [ERROR or No Result]")

if __name__ == "__main__":
    main()
