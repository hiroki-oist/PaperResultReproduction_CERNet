import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def create_perturbation_animation(pred_xyz: np.ndarray,
                                  obs_xyz:  np.ndarray,
                                  future_xyz: Optional[np.ndarray],
                                  error_hs: np.ndarray,
                                  output_errors: np.ndarray,
                                  save_path: str = "perturbation_anim.mp4",
                                  fps: int = 5) -> None:

    T, _      = pred_xyz.shape
    n_layers  = error_hs.shape[1]

    fig, axes = plt.subplots(3, 1, figsize=(8, 11), constrained_layout=True)

    # ---- ① layer-wise hidden error ------------------------------------
    lines_h = [axes[0].plot([], [], label=f"L{l}")[0] for l in range(n_layers)]
    axes[0].set_xlim(0, T)
    axes[0].set_ylim(0, 1.1 * error_hs.max())
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("‖error_h‖")
    axes[0].set_title("Layer-wise hidden-state error")
    axes[0].legend()

    # ---- ② output L2 error --------------------------------------------
    (line_out,) = axes[1].plot([], [], "r-", lw=2)
    axes[1].set_xlim(0, T)
    axes[1].set_ylim(0, 1.1 * output_errors.max())
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("‖pred − obs‖")
    axes[1].set_title("Output error (L2)")

    # ---- ③ XY trajectory ----------------------------------------------
    (line_obs,) = axes[2].plot([], [], "-", label="observation", lw=1.2, alpha=0.5)
    line_future = None
    if future_xyz is not None:
        (line_future,) = axes[2].plot([], [], "-", label="prediction", lw=2.5, alpha=1.0)

    x_vals = np.r_[pred_xyz[:, 0], obs_xyz[:, 0]]
    y_vals = np.r_[-pred_xyz[:, 1], -obs_xyz[:, 1]]
    x_min, x_max = x_vals.min() - .05, x_vals.max() + .05
    y_min, y_max = y_vals.min() - .05, y_vals.max() + .05

    axes[2].set_xlim(y_min, y_max)
    axes[2].set_ylim(x_min, x_max)
    axes[2].set_aspect("equal", "box")
    axes[2].set_xlabel("−y (m)")
    axes[2].set_ylabel("x (m)")
    axes[2].set_title("Trajectory in (−y , x) plane")
    axes[2].legend()
    axes[2].set_yticks([])
    axes[2].set_xticks([])
    axes[2].grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    # ---- 拡大図 inset --------------------------------------------------
    ax_inset = inset_axes(axes[2], width="35%", height="35%", loc="lower right")
    ax_inset.set_aspect("equal", "box")
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])

    x_center = (x_min + x_max) / 2 + (x_max - x_min) / 8
    y_center = (y_min + y_max) / 2 + (y_max - y_min) / 8
    dx = (x_max - x_min) / 8
    dy = (y_max - y_min) / 8

    ax_inset.set_xlim(y_center - dy/2, y_center + dy/2)
    ax_inset.set_ylim(x_center - dx/2, x_center + dx/2)
    mark_inset(axes[2], ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")

    line_obs_inset, = ax_inset.plot([], [], "-", lw=1.2, alpha=0.5)
    line_future_inset = None
    if line_future is not None:
        line_future_inset, = ax_inset.plot([], [], "-", lw=2.5, alpha=1.0)

    # ---- Animation functions ------------------------------------------
    def init():
        for ln in lines_h:
            ln.set_data([], [])
        line_out.set_data([], [])
        line_obs.set_data([], [])
        line_obs_inset.set_data([], [])
        if line_future:
            line_future.set_data([], [])
        if line_future_inset:
            line_future_inset.set_data([], [])

        artists = lines_h + [line_out, line_obs, line_obs_inset]
        if line_future:
            artists.append(line_future)
        if line_future_inset:
            artists.append(line_future_inset)
        return artists

    def update(t):
        xs = np.arange(t + 1)

        for l, ln in enumerate(lines_h):
            ln.set_data(xs, error_hs[:t + 1, l])
        line_out.set_data(xs, output_errors[:t + 1])

        mask_obs = obs_xyz[:t + 1, 2] < -0.38
        line_obs.set_data(-obs_xyz[:t + 1][mask_obs, 1],
                           obs_xyz[:t + 1][mask_obs, 0])
        line_obs_inset.set_data(-obs_xyz[:t + 1][mask_obs, 1],
                                 obs_xyz[:t + 1][mask_obs, 0])

        if line_future is not None:
            fut_t = future_xyz[t]
            if fut_t.size > 0:
                valid = (~np.isnan(fut_t[:, 0])) & (fut_t[:, 2] < -0.38)
                line_future.set_data(-fut_t[valid, 1], fut_t[valid, 0])
                if line_future_inset:
                    line_future_inset.set_data(-fut_t[valid, 1], fut_t[valid, 0])

        artists = lines_h + [line_out, line_obs, line_obs_inset]
        if line_future:
            artists.append(line_future)
        if line_future_inset:
            artists.append(line_future_inset)
        return artists

    ani = FuncAnimation(fig, update, frames=T, init_func=init,
                        blit=True, interval=1000 // fps)
    ani.save(save_path, fps=fps, dpi=200)
    plt.close(fig)
    print(f"[INFO] animation saved → {save_path}")
