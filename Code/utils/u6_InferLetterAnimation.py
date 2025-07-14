# ── 9. Animation -----------------------------------------------------
# (unchanged except for variable names) --------------------------------

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import gridspec
import numpy as np
import os 

def makeAnimationInferLetter(obs_buf, pred_buf, seq_len, mse_buf, c_buf, past_reconstruction_buf, save_root, tag):

    # ╭──────────────────── utility ─────────────────────────────╮
    def _lims(arr, margin=0.05):
        lo, hi = arr.min(), arr.max()
        pad = (hi - lo) * margin if hi > lo else 0.01
        return lo - pad, hi + pad

    # Axis ranges
    all_x = np.concatenate([obs_buf[:, 0], pred_buf[:, 0].ravel()])
    all_y = np.concatenate([obs_buf[:, 1], pred_buf[:, 1].ravel()])
    x_lo, x_hi = _lims(all_x)
    y_lo, y_hi = _lims(all_y)

    neg_all_y = -all_y
    x_all = all_x
    x2_lo, x2_hi = _lims(neg_all_y)      # horizontal = -Y
    y2_lo, y2_hi = _lims(x_all)          # vertical   =  X

    T_TOTAL = seq_len

    # ╭──────────────────── Figure / GridSpec ─────────────────────────╮
    fig = plt.figure(figsize=(12, 9))
    gs  = gridspec.GridSpec(
            nrows=6, ncols=3,
            height_ratios=[1, 1, 1, 1.5, 1.5, .1],
            hspace=0.5, wspace=0.35
        )

    # 1–3 rows: X, Y, MSE (full width)
    ax_x   = fig.add_subplot(gs[0, :])
    ax_y   = fig.add_subplot(gs[1, :])
    ax_mse = fig.add_subplot(gs[2, :])

    for ax, lbl, ylim in zip((ax_x, ax_y),
                            ('X [m]', 'Y [m]'),
                            ((x_lo, x_hi), (y_lo, y_hi))):
        ax.set_ylabel(lbl)
        ax.set_xlim(-1, T_TOTAL)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
    ax_y.set_xlabel('Time step')

    ax_mse.set_ylabel('MSE')
    ax_mse.set_xlim(-1, seq_len)
    ax_mse.set_ylim(0, mse_buf.max()*1.05 + 1e-9)
    ax_mse.grid(True, alpha=0.3)
    ax_mse.set_xlabel('Time step')

    # 4th row (3 panels): Obs / Recon / c‑hist
    ax_xy_obs   = fig.add_subplot(gs[3, 0])
    ax_xy_recon = fig.add_subplot(gs[3, 1])
    ax_c_hist   = fig.add_subplot(gs[3, 2])

    for ax, ttl in zip((ax_xy_obs, ax_xy_recon),
                    ('Observed', 'Reconstructed')):
        ax.set_title(ttl)
        ax.set_xlabel('-Y [m]'); ax.set_ylabel('X [m]')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.set_xlim(x2_lo, x2_hi); ax.set_ylim(y2_lo, y2_hi)

    # c bar graph initialisation
    c_dim = c_buf.shape[1]          # 26
    bars  = ax_c_hist.bar(np.arange(c_dim), np.zeros(c_dim),
                        color='gray')
    ax_c_hist.set_ylim(-1.1, 1.1)
    ax_c_hist.set_xlabel('class‑id'); ax_c_hist.set_ylabel('c value')
    ax_c_hist.set_xticks(np.arange(c_dim))
    ax_c_hist.set_title('c (max=red  2nd=orange)')

    # 5th row: c heat‑map (full‑width)
    ax_c = fig.add_subplot(gs[4, :])
    im_c = ax_c.imshow(np.full_like(c_buf.T, np.nan),
                    origin='lower', aspect='auto',
                    cmap='viridis', vmin=0, vmax=1)
    ax_c.set_ylabel('class‑id')
    ax_c.set_xlabel('time step')
    ax_c.set_yticks(np.arange(c_dim))
    ax_c.set_ylim(-.5, c_dim - 0.5)

    # Draw‑ables ---------------------------------------------------------
    obs_line_x,   = ax_x.plot([], [], 'b-', label='Obs x')
    obs_line_y,   = ax_y.plot([], [], 'b-', label='Obs y')
    recon_line_x, = ax_x.plot([], [], 'c--', lw=1, label='Recon x')
    recon_line_y, = ax_y.plot([], [], 'c--', lw=1, label='Recon y')
    mse_line,     = ax_mse.plot([], [], 'g-', label='MSE')

    obs_traj_line,   = ax_xy_obs.plot([], [], 'b-')
    recon_traj_line, = ax_xy_recon.plot([], [], 'c--')

    for ax in (ax_x, ax_y, ax_mse, ax_xy_obs, ax_xy_recon):
        ax.legend(loc='upper left', fontsize=8)

    # init / update functions -------------------------------------------
    def _init():
        for ln in (obs_line_x, obs_line_y, recon_line_x, recon_line_y,
                    mse_line, obs_traj_line, recon_traj_line):
            ln.set_data([], [])
        im_c.set_data(np.full_like(c_buf.T, np.nan))
        for b in bars: b.set_height(0); b.set_color('gray')
        return (*bars, obs_line_x, obs_line_y,
                recon_line_x, recon_line_y, mse_line,
                obs_traj_line, recon_traj_line, im_c)

    def _update(frame):
        t_axis = np.arange(frame + 1)

        # time series ----------------------------------------------------
        obs_line_x.set_data(t_axis, obs_buf[:frame+1, 0])
        obs_line_y.set_data(t_axis, obs_buf[:frame+1, 1])

        recon_x = past_reconstruction_buf[frame, :, 0]
        recon_y = past_reconstruction_buf[frame, :, 1]
        recon_t_axis = np.arange(seq_len)
        recon_line_x.set_data(recon_t_axis, recon_x)
        recon_line_y.set_data(recon_t_axis, recon_y)

        mse_line.set_data(t_axis, mse_buf[:frame+1])
        ax_mse.set_ylim(0, mse_buf[:frame+1].max()*1.05 + 1e-9)

        # XY trajectories -----------------------------------------------
        obs_traj_line.set_data(-obs_buf[:frame+1, 1], obs_buf[:frame+1, 0])
        recon_traj_line.set_data(-recon_y, recon_x)

        # c histogram ----------------------------------------------------
        cur_c = c_buf[frame]
        top2  = np.argsort(cur_c)[-2:]
        for i, b in enumerate(bars):
            b.set_height(cur_c[i])
            b.set_color('orange' if i == top2[0] else
                        'red'     if i == top2[1] else
                        'gray')

        # c heat‑map ------------------------------------------------------
        c_img = np.full_like(c_buf.T, np.nan)
        c_img[:, :frame+1] = c_buf[:frame+1].T
        im_c.set_data(c_img)

        return (*bars, obs_line_x, obs_line_y,
                recon_line_x, recon_line_y, mse_line,
                obs_traj_line, recon_traj_line, im_c)

    # Animation ---------------------------------------------------------
    ani = FuncAnimation(fig, _update, frames=seq_len,
                        init_func=_init, blit=True)
    video_path = os.path.join(save_root, f"prediction_anim_{tag}.mp4")
    ani.save(video_path, writer=FFMpegWriter(fps=5))
    print(f"[INFO] animation saved → {video_path}")
    
def makePaperFigureInferLetter(obs_buf, past_reconstruction_buf, c_buf, save_path=None):
    """
    Create a figure for paper visualization:
    - Top row: 5 subplots (t=10, 30, 50, 70, 90) showing predicted vs observed trajectories
      using only points where Z < -0.38
    - Bottom row: 26 class embeddings over time as line plots (values < 0 treated as 0)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    timesteps = [10, 30, 50, 70, 90]
    num_plots = len(timesteps)
    c_dim = c_buf.shape[1]
    seq_len = c_buf.shape[0]
    z_threshold = -0.38

    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(nrows=2, ncols=num_plots, height_ratios=[2, 1], hspace=0.4)

    # --- Compute common X/Y limits across all filtered data ---
    all_x, all_y = [], []
    for t in timesteps:
        obs = obs_buf[:t+1]
        obs_filt = obs[obs[:, 2] < z_threshold]
        recon = past_reconstruction_buf[t]
        recon_filt = recon[recon[:, 2] < z_threshold]
        all_x.extend(-obs_filt[:, 1]); all_x.extend(-recon_filt[:, 1])
        all_y.extend(obs_filt[:, 0]);  all_y.extend(recon_filt[:, 0])

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # --- Top row: filtered trajectory plots ---
    for i, t in enumerate(timesteps):
        ax = fig.add_subplot(gs[0, i])
        obs = obs_buf[:t+1]
        obs_filt = obs[obs[:, 2] < z_threshold]
        recon = past_reconstruction_buf[t]
        recon_filt = recon[recon[:, 2] < z_threshold]

        ax.plot(-obs_filt[:, 1], obs_filt[:, 0], 'b-', label='Observed', linewidth=2)
        ax.plot(-recon_filt[:, 1], recon_filt[:, 0], 'c--', label='Predicted', linewidth=2)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Timestep {t}', fontsize=22)
        ax.set_xlabel(''); ax.set_ylabel('')
        ax.tick_params(axis='both', which='both', direction='in', length=4,
                       labelbottom=False, labelleft=False)
        if i == 0:
            ax.legend(
                fontsize=16,
                loc='lower left',
                bbox_to_anchor=(-0.2, 0.05) 
            )
            
    # --- Bottom row: class embedding line plot ---
    ax_line = fig.add_subplot(gs[1, :])
    t_axis = np.arange(seq_len)
    c_clipped = np.clip(c_buf, 0, 1)  # clamp negatives to 0

    colors = plt.cm.viridis(np.linspace(0, 1, c_dim))

    highlight_indices = {1: 'b', 5: 'f', 16: 'q'}  # index: legend label
    for i in range(c_dim):
        if i in highlight_indices:
            ax_line.plot(
                t_axis,
                c_clipped[:, i],
                label=highlight_indices[i],
                color=colors[i],
                linewidth=3.5,
                alpha=1.0,
            )
        else:
            ax_line.plot(
                t_axis,
                c_clipped[:, i],
                color=colors[i],
                linewidth=1,
                alpha=0.3,
            )

    ax_line.set_title('Class Embedding Vector $\\mathbf{C}$ Over Time', fontsize=22)
    ax_line.set_xlabel('Timestep', fontsize=22)
    ax_line.set_ylabel('Value', fontsize=22)
    ax_line.set_xlim(0, seq_len - 1)
    ax_line.set_ylim(0, 0.5)
    ax_line.grid(True, alpha=0.3)
    ax_line.tick_params(labelsize=12)
    # ax_line.legend(loc='upper right', ncol=1, fontsize=18)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[INFO] Figure saved to {save_path}")
    else:
        plt.show()
