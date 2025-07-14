#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from dtaidistance import dtw
import matplotlib.pyplot as plt

from utils.x1_path_utils import resolve_path

# ────────────────────────── Path ──────────────────────────
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

DATA_ROOT  = PROJECT_ROOT / "Data" / "exp1_SequenceGeneration"
TRAIN_PATH = "Data/train_data_26_100_3.npy"

# ────────────────────────── Model to be evaluated ──────────────────────────
MODELS = [
     "SingleStandard", "SingleLarge", "SingleMini",
     "MultiStandard",  "MultiLarge", "MultiMini"
]
SEEDS      = range(1, 11)
CLASS_IDS  = range(26)

# ────────────────────────── Compute DTW ──────────────────────────
print("[INFO] Loading training dataset …")
train_data = np.load(resolve_path(TRAIN_PATH))          # shape = (26, 100, 3)

results             = {}  # {model: (mean, std)}
best_seed_per_model = {}  # {model: (best_seed, best_mean)}

for model in MODELS:
    print(f"\n[MODEL] {model}")
    per_seed_means = {}
    all_dtw = []

    for seed in SEEDS:
        seed_dir = DATA_ROOT / model / f"Seed{seed}"
        if not seed_dir.exists():
            print(f"  [WARN] Missing: {seed_dir}")
            continue

        obs_seq = []
        for cid in CLASS_IDS:
            npy_path = seed_dir / f"obs_class_id_{cid}.npy"
            if not npy_path.exists():
                print(f"  [WARN] Missing: {npy_path.name}")
                break
            obs_seq.append(np.load(npy_path))

        if len(obs_seq) != len(CLASS_IDS):
            continue  # skip incomplete sets

        obs_seq = np.stack(obs_seq)       # (26, 100, 3)
        dtws = [
            dtw.distance(train_data[i, :, j], obs_seq[i, :, j])
            for i in range(26) for j in range(3)
        ]

        per_seed_means[seed] = np.mean(dtws)
        all_dtw.extend(dtws)

    if all_dtw:
        mean_dtw, std_dtw = np.mean(all_dtw), np.std(all_dtw)
        results[model] = (mean_dtw, std_dtw)
        best_seed = min(per_seed_means, key=per_seed_means.get)
        best_seed_per_model[model] = (best_seed, per_seed_means[best_seed])
        print(f"  → DTW mean: {mean_dtw:.6f}, std: {std_dtw:.6f} (n={len(all_dtw)})")
        print(f"  → Best seed: {best_seed}  (mean DTW = {per_seed_means[best_seed]:.6f})")
    else:
        print("  → No valid DTW computed.")

print("\n[SUMMARY]")
for model, (mean, std) in results.items():
    print(f"{model:<12} DTW mean = {mean:.6f} | std = {std:.6f}")

print("\n[BEST SEED PER MODEL]")
for model, (seed, best_mean) in best_seed_per_model.items():
    print(f"{model:<12} best Seed = {seed:<2d} | mean DTW = {best_mean:.6f}")

# ────────────────────────── Setting for plots ──────────────────────────
CLASS_IDS_TO_PLOT = [1, 4, 10, 11, 12]
Z_THRESHOLD       = -0.38
OFFSET_X          = 0.2
FIGSIZE           = (12, 4.8)
TRAIN_COLOR       = "navy"
OBS_COLOR         = "darkorange"

SIZE_ORDER = ["Mini", "Standard", "Large"]  
COL_ORDER  = ["Single", "Multi"]           

row_cats = [s for s in SIZE_ORDER if any(s in m for m in MODELS)]
col_cats = [c for c in COL_ORDER  if any(m.startswith(c) for m in MODELS)]
n_rows, n_cols = len(row_cats), len(col_cats)

fig, ax_arr = plt.subplots(
    n_rows, n_cols,
    figsize=FIGSIZE,
    gridspec_kw=dict(hspace=0.01, wspace=0.05)
)
axs = np.atleast_2d(ax_arr).reshape(n_rows, n_cols)

def _subplot_pos(name: str):
    size   = next(s for s in SIZE_ORDER if s in name)
    prefix = "Single" if name.startswith("Single") else "Multi"
    return row_cats.index(size), col_cats.index(prefix)

legend_handles, legend_labels = [], []

for model in MODELS:
    if model not in best_seed_per_model:   
        continue
    seed, _ = best_seed_per_model[model]
    seed_dir = DATA_ROOT / model / f"Seed{seed}"
    row, col = _subplot_pos(model)
    ax = axs[row, col]

    for idx, cid in enumerate(CLASS_IDS_TO_PLOT):
        offset = np.array([0.0, idx * OFFSET_X])

        # ── Train (baseline) -------------------------------------------------
        xy_t = train_data[cid][train_data[cid][:, 2] < Z_THRESHOLD, :2]
        xy_t[:, 1] *= -1            
        xy_t += offset
        line_t, = ax.plot(
            xy_t[:, 1], xy_t[:, 0],
            linestyle="--", linewidth=1, color=TRAIN_COLOR,
            label="Train (baseline)" if not legend_handles else None
        )

        # ── Obs (predicted) --------------------------------------------------
        obs_path = seed_dir / f"obs_class_id_{cid}.npy"
        if obs_path.exists():
            xy_o = np.load(obs_path)
            xy_o = xy_o[xy_o[:, 2] < Z_THRESHOLD, :2]
            xy_o[:, 1] *= -1
            xy_o += offset
            line_o, = ax.plot(
                xy_o[:, 1], xy_o[:, 0],
                linewidth=1.5, color=OBS_COLOR,
                label="Obs (predicted)" if not legend_handles else None
            )

            if not legend_handles:
                legend_handles += [line_t, line_o]
                legend_labels  += ["Train Data", "Model Output"]

    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)

for col, prefix in enumerate(col_cats):
    axs[0, col].set_title(prefix, fontsize=14, pad=8)

for r, label in enumerate(row_cats):
    axs[r, 0].text(
        -0.03, 0.5, label,
        transform=axs[r, 0].transAxes,
        ha="right", va="center", fontsize=14
    )

if legend_handles:
    bbox_x = 0.92 if n_cols == 1 else 0.90
    plt.subplots_adjust(left=0.15, top=0.85) 
    fig.legend(
        legend_handles, legend_labels,
        loc="upper center", ncol=1,
        bbox_to_anchor=(bbox_x, 0.94),
        frameon=True, fancybox=False,
        edgecolor="black", facecolor="white",
        framealpha=1.0, fontsize=14
    )

plt.show()
