import os
import time
import argparse
import pickle as pk
import json
import sys
from datetime import datetime
import warnings

import numpy as np
import torch

from dotenv import load_dotenv
load_dotenv()

from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode

from model.CERNet import CERNet
from utils import u2_tomlConfig as toml_config
from utils.u6_InferLetterAnimation import makeAnimationInferLetter, makePaperFigureInferLetter
from utils.x1_path_utils import resolve_path

SAFE_MIN, SAFE_MAX = -0.8, 0.8   # not used at the moment
HORIZON = 20                      # reserve for future improvements

# ╭────────────────── utils ─────────────────────────────────────────────╮

def normalise(x: np.ndarray, dmin: float, dmax: float) -> np.ndarray:
    """Scale raw data in metres → [-0.9, 0.9]."""
    return -0.9 + (x - dmin) * 1.8 / (dmax - dmin)


def denormalise(xn: np.ndarray, dmin: float, dmax: float) -> np.ndarray:
    """Scale back from [-0.9, 0.9] → metres."""
    return ((xn + 0.9) / 1.8) * (dmax - dmin) + dmin


# ╭──────────────── snapshot loader ─────────────────────────────────────╮

def _load_pcrnn_snapshot(model: CERNet, snap_path: str):
    with open(snap_path, 'rb') as f:
        ss = pk.load(f)
    model.w_o.copy_(torch.from_numpy(ss['w_o']))
    model.b_o.copy_(torch.from_numpy(ss['b_o']))
    model.w_c.copy_(torch.from_numpy(ss['w_c']))
    for i, arr in enumerate(ss['w_r']):
        model.w_r[i].data.copy_(torch.from_numpy(arr))
    for i, arr in enumerate(ss['b_r']):
        model.b_r[i].data.copy_(torch.from_numpy(arr))
    for i, arr in enumerate(ss['w_hh']):
        model.w_hh[i].data.copy_(torch.from_numpy(arr))
    print(f"[INFO] snapshot loaded from {snap_path}")


# ╭──────────────── model builder ───────────────────────────────────────╮

def build_model(cfg, ocfg, seq_id, rand_seed):
    mtype = ocfg['model_type'].lower()
    if mtype == 'pcrnn':
        net = CERNet(
            causes_dim=cfg['training']['causes_dim'],
            states_dim=cfg['training']['states_dim'],
            output_dim=cfg['training']['output_dim'],
            tau_h     =cfg['training']['tau_h'],
            alpha_x   =cfg['inferLetter']['alpha_x'],
            alpha_h   =cfg['inferLetter']['alpha_h'])
        path = resolve_path(ocfg['model_path'])
        if str(path).endswith('.pkl'):
            _load_pcrnn_snapshot(net, path)
        else:
            net.load_state_dict(torch.load(path, map_location='cpu'))
    else:
        raise ValueError(f'unknown model_type "{mtype}"')

    # Reset hidden state. We keep class_id=None so that PC‑RNN decides on its own.
    net.eval(); net.reset_state(batch_size=1, class_id=None, rand_seed=rand_seed)
    return net, mtype


def move_to_xyz(reachy, xyz: np.ndarray, grip_deg: float, duration: float = 1.0):
    """Move Reachy’s left arm end‑effector to *xyz* (m) with a fixed orientation."""
    x, y, z = xyz
    tf = np.array([[-0.029, 0.284, -0.959, x],
                   [ 0.024, 0.959,  0.283, y],
                   [ 0.999,-0.015, -0.035, z],
                   [ 0.   , 0.   ,  0.   , 1.]])
    q = reachy.l_arm.inverse_kinematics(tf)

    reachy.turn_on('l_arm')
    goto({j: p for j, p in zip(reachy.l_arm.joints.values(), q)},
         duration=duration, interpolation_mode=InterpolationMode.MINIMUM_JERK)

    if hasattr(reachy.l_arm, 'l_gripper'):
        reachy.l_arm.l_gripper.goal_position = grip_deg


# ╭──────────────── main runner ─────────────────────────────────────────╮

def run_online(toml_path: str, seq_id: int, rand_seed: int, animation: bool = False,  use_robot: bool = True):
    """Run a single online inference session.

    Args:
        toml_path (str): Path to the TOML config file.
        seq_id (int): Sequence ID to run. Must be within dataset range.
        use_robot (bool): Whether to control the physical Reachy.
    """
    cfg  = toml_config.TOMLConfigLoader(toml_path)
    ocfg = cfg['onlineER']

    # ── 1. Load training dataset to reproduce dmin/dmax ─────────────────
    data = np.load(resolve_path(cfg['dataset']['dataset_path']))  # raw metres, shape (N, T, 3)
    if data.ndim != 3:
        raise RuntimeError('Dataset must be 3‑dimensional: (Nseq, T, 3)')
    n_seq, seq_len, _ = data.shape

    assert 0 <= seq_id < n_seq, (
        f'seq_id {seq_id} out of range (0 .. {n_seq - 1})')
    print(f"[SEQ] using sequence id {seq_id}")

    dmin, dmax = data.min(), data.max()

    gt_xyz = data[seq_id]            # (T, 3)
    init_xyz = gt_xyz[0].copy()
        
    if use_robot:
        print("Connecting to Reachy")
        reachy = ReachySDK(host='10.42.0.1')
        move_to_xyz(reachy, init_xyz, ocfg['gripper_deg'])
    else:
        reachy = None

    # ── 3. Model ─────────────────────────────────────────────────────────
    net, mtype = build_model(cfg, ocfg, seq_id, rand_seed)
    fb_flag    = ocfg.get('use_feedback', True) and (mtype == 'pcrnn')

    # ── 5. Buffers & timing ────────────────────────────────────────────
    pred_buf = np.zeros((seq_len, 3))
    obs_buf  = np.zeros((seq_len, 3))
    dt       = 1.0 / 10  # fixed to 10 FPS

    error_buf = np.zeros((seq_len, 3))
    pred_prev_xyz = init_xyz.copy()  # seed for sim‑mode
    obs_xyz = np.zeros(3)

    c_buf = np.zeros((seq_len, 26))
    past_reconstruction_buf = np.zeros((seq_len, seq_len, 3))
    mse_buf = np.zeros((seq_len))

    print(f"[RUN] {mtype.upper()} | T={seq_len} | fb={fb_flag} | robot={use_robot}")

    # ── 6. Main loop ────────────────────────────────────────────────────
    for t in range(seq_len):
        tic = time.time()

        pred_prev_norm = normalise(pred_prev_xyz, dmin, dmax)
        pred_prev_ts   = torch.tensor([pred_prev_norm], dtype=torch.float32)

        # Predict output for time‑step t
        computation_start = time.time()
        if t == 0:  # at t==0, no observation yet
            mse_sum, preds = net.detect_class(t, None, feedback=True)
            pred_norm = preds[0]
        else:
            mse_sum, preds = net.detect_class(t, obs_ts, feedback=True)
            pred_norm = preds[t]
        computation_end = time.time()
        computation_duration = computation_end - computation_start
        print("Model Computation Time: ", computation_duration)

        pred_xyz = denormalise(pred_norm, dmin, dmax)
        pred_buf[t] = pred_xyz
        pred_prev_xyz = pred_xyz
        
        if t > 50 and use_robot:
            fk = reachy.l_arm.forward_kinematics()
            obs_xyz = fk[:3, 3].copy()
        else:
            obs_xyz = gt_xyz[t]
            
        obs_buf[t] = obs_xyz
        obs_norm = normalise(obs_xyz, dmin, dmax)
        obs_ts   = torch.tensor([obs_norm], dtype=torch.float32)

        error_buf[t] = pred_xyz - obs_xyz
        c_buf[t] = net.c.squeeze(0).numpy()
        if t > 0:
            mse_buf[t] = mse_sum / t
        past_reconstruction_buf[t] = denormalise(preds, dmin, dmax)
        
        if use_robot:
            # 安全停止コード
            if (pred_prev_xyz < SAFE_MIN).any() or (pred_prev_xyz > SAFE_MAX).any() \
            or (obs_xyz       < SAFE_MIN).any() or (obs_xyz       > SAFE_MAX).any() \
            or (pred_xyz      < SAFE_MIN).any() or (pred_xyz      > SAFE_MAX).any():
                msg = (f"[ABORT] Value outside [{SAFE_MIN}, {SAFE_MAX}] m detected at t={t}\n"
                    f"  pred_prev: {pred_prev_xyz}\n"
                    f"  obs      : {obs_xyz}\n"
                    f"  pred_next: {pred_xyz}")
                print(msg, file=sys.stderr)
                if use_robot:
                    reachy.turn_off_smoothly('l_arm')
                raise RuntimeError(msg) 
            
            # 6‑3) send to robot ---------------------------------------------
            tf = np.array([[-0.029, 0.284,-0.959, pred_xyz[0]],
                           [ 0.024, 0.959, 0.283, pred_xyz[1]],
                           [ 0.999,-0.015,-0.035, pred_xyz[2]],
                           [ 0.   , 0.   , 0.   , 1.]])
            try:
                q = reachy.l_arm.inverse_kinematics(tf)
                start = time.time()
                D_MIN = 0.02                # goto が受け付ける最短 20 ms   (sampling_freq=100Hz 前提)
                duration_cmd = max(D_MIN, dt - computation_duration)
                goto({j: p for j, p in zip(reachy.l_arm.joints.values(), q)},
                     duration=duration_cmd, interpolation_mode=InterpolationMode.MINIMUM_JERK)
                end = time.time()
                print("Goto took: ", end - start)
            except Exception as e:
                print(f"[WARN] IK fail @t={t}: {e}")
            if hasattr(reachy.l_arm, 'l_gripper'):
                reachy.l_arm.l_gripper.goal_position = ocfg['gripper_deg']

        # Debug printout --------------------------------------------------
        print("--------------")
        print(
            "step:", t,
            "\nprediction:", pred_prev_xyz,
            "\nobservation:", obs_xyz,
            "\nerror:", pred_prev_xyz - obs_xyz,
        )

        # Keep FPS --------------------------------------------------------
        elapsed   = time.time() - tic
        print("elapsed: ", elapsed)
        remaining = dt - elapsed

        if remaining > 0:
            time.sleep(remaining)
        else:
            # 遅れが発生したので警告だけ出す（処理は続行）
            warnings.warn(
                f"Step over-run by {-remaining:.4f} s (elapsed = {elapsed:.4f}s, dt = {dt:.4f}s)",
                RuntimeWarning,
                stacklevel=1
            )

    print("[DONE] sequence finished")
    
    if use_robot:
        reachy.turn_off_smoothly('l_arm')

    # ── 7. Error statistics ────────────────────────────────────────────
    error_mean = error_buf.mean(axis=0)
    error_cov  = np.cov(error_buf.T)
    error_l2   = np.linalg.norm(error_buf, axis=1)

    print("\n[ERROR STATS]")
    print(f"  mean error vector     : {error_mean} (m)")
    print(f"  covariance matrix (m²):\n{error_cov}")
    print(f"  L2 norm mean          : {error_l2.mean():.6f} m")
    print(f"  L2 norm std.dev       : {error_l2.std():.6f} m")
    print(f"  L2 norm 95‑percentile : {np.quantile(error_l2, 0.95):.6f} m")

    # ── 8. Save trajectories & metadata ────────────────────────────────
    save_root = resolve_path(ocfg['inferLetter_save_directory'])
    os.makedirs(save_root, exist_ok=True)
    tag = f"class{seq_id:02d}_" + datetime.now().strftime('%Y%m%d_%H%M%S')

    np.save(os.path.join(save_root, f'pred_xyz_{tag}.npy'), pred_buf)
    np.save(os.path.join(save_root, f'obs_xyz_{tag}.npy'),  obs_buf)
    print(f"[INFO] trajectories saved → {save_root}")

    # --- 8‑b) Recognition metadata -------------------------------------
    final_c = c_buf[seq_len - 1]  # last time‑step
    top2_idx = np.argsort(final_c)[-2:]  # ascending order
    top1, top2 = int(top2_idx[1]), int(top2_idx[0])

    meta = {
        'seq_id': int(seq_id),
        'top1': top1,
        'top2': top2,
        'match_top1': seq_id == top1,
        'match_top2': seq_id == top2,
        'success': seq_id in (top1, top2),
        'final_mse': mse_buf[-1],
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }

    meta_path = os.path.join(save_root, f'meta_{tag}.json')
    with open(meta_path, 'w', encoding='utf‑8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[INFO] metadata saved → {meta_path}")

    if animation:
        makeAnimationInferLetter(obs_buf, pred_buf, seq_len, mse_buf, c_buf, past_reconstruction_buf, save_root, tag)
        makePaperFigureInferLetter(obs_buf, past_reconstruction_buf, c_buf, save_path=None)

# ╭────────────────────────── CLI ───────────────────────────────────────╮
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='Path to TOML configuration')
    p.add_argument('--animation', type=bool, default=False, help='Make animation (default False)')
    p.add_argument('--rand_seed', type=int, default=1, help='random seed')

    try:
        p.add_argument('--robot', action=argparse.BooleanOptionalAction, default=False,
                       help='Control physical Reachy (default True). "--no-robot" for pure simulation.')
    except AttributeError:
        # Python <3.9 fallback
        def _str2bool(v):
            if isinstance(v, bool):
                return v
            if v.lower() in {'yes', 'true', 't', 'y', '1'}:
                return True
            if v.lower() in {'no', 'false', 'f', 'n', '0'}:
                return False
            raise argparse.ArgumentTypeError('Boolean value expected.')
        p.add_argument('--robot', type=_str2bool, default=True)

    p.add_argument('--seq-id', type=int, required=True,
                   help='Sequence ID to use for inference (0‑based index).')

    args = p.parse_args()

    run_online(args.config, args.seq_id, args.rand_seed, args.animation, use_robot=args.robot)
