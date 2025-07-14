#!/usr/bin/env python3
import argparse, glob, os, re
import numpy as np
import matplotlib.pyplot as plt

Z_TH      = -0.38            # plot only z ≤ -0.38 (table height)
SAVE_NAME = "evaluate_summary.png"

from pathlib import Path
import glob, os, re, numpy as np

def _collect(run_dir: str):
    """Load corresponding *_xyz / *_hs / *_err files in a robust way.

    A tag is everything between the common prefix (e.g. 'obs_xyz_')
    and the file extension '.npy'.  This works no matter how many
    digits the class ID contains.
    """
    # ---- build tag list -------------------------------------------------
    obs_files = glob.glob(os.path.join(run_dir, "obs_xyz_*.npy"))
    tags = [Path(f).stem[len("obs_xyz_"):] for f in obs_files]  # e.g. 'class01_20250714_175335'
    tags.sort()

    # ---- load arrays ----------------------------------------------------
    obs_list, err_h_list, out_err_list = [], [], []
    for tag in tags:
        obs_list.append(np.load(os.path.join(run_dir, f"obs_xyz_{tag}.npy")))
        err_h_list.append(np.load(os.path.join(run_dir, f"error_hs_{tag}.npy")))
        out_err_list.append(np.load(os.path.join(run_dir, f"out_err_{tag}.npy")))

    return np.stack(obs_list), np.stack(err_h_list), np.stack(out_err_list)



def _plot_summary(obs_list, err_h_list, out_err_list):
    n_runs = len(obs_list)
    
    plt.rcParams.update({
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14
    })

    fig, axs = plt.subplot_mosaic(
        [["err_h",  "traj"],
         ["out_err","traj"]],
        figsize=(11, 6),
        constrained_layout=True
    )
    ax_h, ax_out, ax_traj = axs["err_h"], axs["out_err"], axs["traj"]

    # ---- 1) layer-wise hidden error ----------------------------------
    for e in err_h_list:
        ax_h.plot(np.linalg.norm(e, axis=1), alpha=.25, lw=1, color="tab:blue")
    mean_h = np.mean([np.linalg.norm(e, axis=1) for e in err_h_list], axis=0)
    ax_h.plot(mean_h, color="tab:blue", lw=2.5, label="mean")
    ax_h.set_ylabel("‖error_h‖")
    ax_h.set_xlabel("t")
    ax_h.set_title("Layer-wise hidden error (L2 norm)")
    ax_h.legend()

    # ---- 2) output L2 error ------------------------------------------
    for o in out_err_list:
        ax_out.plot(o, alpha=.25, lw=1, color="tab:red")
    mean_out = np.mean(out_err_list, axis=0)
    ax_out.plot(mean_out, color="tab:red", lw=2.5, label="mean")
    ax_out.set_ylabel("‖pred − obs‖")
    ax_out.set_xlabel("t")
    ax_out.set_title("Output error (L2 norm)")
    ax_out.legend()

    # ---- 3) XY trajectory --------------------------------------------
    # mask = baseline[:, 2] <= Z_TH
    # ax_traj.plot(-baseline[mask, 1], baseline[mask, 0],
    #              color="k", lw=3, label="baseline")

    for obs in obs_list:
        m = obs[:, 2] <= Z_TH
        ax_traj.plot(-obs[m, 1], obs[m, 0],
                     lw=1.2, alpha=.3, color="tab:green")

    ax_traj.set_aspect("equal", "box")
    ax_traj.set_xlabel("−y (m)")
    ax_traj.set_ylabel("x (m)")
    ax_traj.set_title(f"Trajectory of Reachy — {n_runs} runs")
    ax_traj.legend()

    fig.savefig(SAVE_NAME, dpi=200)
    print(f"[INFO] saved → {SAVE_NAME}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True,
                    help="perturbation_save_directory (10 run files が入っているフォルダ)")
    args = ap.parse_args()

    # obs_list, err_h_list, out_err_list, baseline = _collect(args.result_dir)
    obs_list, err_h_list, out_err_list = _collect(args.result_dir)
    # if len(obs_list) == 0:
        # raise RuntimeError("No obs_xyz_*.npy found in the given directory")
    _plot_summary(obs_list, err_h_list, out_err_list)


if __name__ == "__main__":
    main()
