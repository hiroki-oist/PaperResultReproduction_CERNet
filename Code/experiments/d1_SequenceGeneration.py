import os, time, argparse, pickle as pk
import numpy as np
import torch
import sys
from dotenv import load_dotenv
load_dotenv()

from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode

from model.CERNet import CERNet
from utils           import u2_tomlConfig as toml_config
from utils.x1_path_utils import resolve_path

SAFE_MIN, SAFE_MAX = -0.8, 0.8      

# ╭────────────────── utils ─────────────────────────────────────────────╮

def normalise(x: np.ndarray, dmin: float, dmax: float) -> np.ndarray:
    """Scale raw data in metres → [-0.9, 0.9]."""
    return -0.9 + (x - dmin) * 1.8 / (dmax - dmin)

def denormalise(xn: np.ndarray, dmin: float, dmax: float) -> np.ndarray:
    """Scale back from [-0.9, 0.9] → metres."""
    return ((xn + 0.9) / 1.8) * (dmax - dmin) + dmin


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


# ╭──────────────── snapshot loader ─────────────────────────────────────╮

def _load_model_snapshot(model: CERNet, snap_path: str):
    with open(snap_path, 'rb') as f:
        ss = pk.load(f)
        
    print("[DEBUG] Snapshot shapes:")
    print("w_o", ss['w_o'].shape, "model.w_o", tuple(model.w_o.shape))
    print("b_o", ss['b_o'].shape, "model.b_o", tuple(model.b_o.shape))
    print("w_c", ss['w_c'].shape, "model.w_c", tuple(model.w_c.shape))
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

def build_model(cfg, ocfg):
    mtype = ocfg['model_type'].lower()
    if mtype == 'pcrnn':
        net = CERNet(
            causes_dim=cfg['training']['causes_dim'],
            states_dim=cfg['training']['states_dim'],
            output_dim=cfg['training']['output_dim'],
            tau_h     =cfg['training']['tau_h'],
            alpha_x   =cfg['onlineER']['alpha_x'],
            alpha_h   =cfg['onlineER']['alpha_h'])
        path = resolve_path(ocfg['model_path'])
        print('Path: ', path)
        if str(path).endswith('.pkl'):
            _load_model_snapshot(net, path)
        else:
            net.load_state_dict(torch.load(path, map_location='cpu'))
    else:
        raise ValueError(f'unknown model_type "{mtype}"')

    net.eval(); net.reset_state(batch_size=1, class_id=ocfg['class_id'])
    return net, mtype


# ╭──────────────── main runner ─────────────────────────────────────────╮

def run_online(toml_path: str, use_robot: bool = True):
    cfg  = toml_config.TOMLConfigLoader(toml_path)
    ocfg = cfg['onlineER']

    # ── 1. Load training dataset to reproduce dmin/dmax ─────────────────
    data = np.load(resolve_path(cfg['dataset']['dataset_path']))  # raw metres
    dmin, dmax = data.min(), data.max()

    # ── 2. Sequence length & initial pose ───────────────────────────────
    if data.ndim == 3:
        n_seq, seq_len, _ = data.shape
        cid = ocfg['class_id']
        assert 0 <= cid < n_seq, f'class_id {cid} out of range (0 - {n_seq-1})'
        init_xyz = data[cid, 0].copy()
    else:
        seq_len = data.shape[0]
        init_xyz = data[0].copy()

    # ── 3. Model ─────────────────────────────────────────────────────────
    net, mtype = build_model(cfg, ocfg)
    fb_flag    = ocfg.get('use_feedback', True) and (mtype == 'pcrnn')

    # ── 4. Robot I/F ────────────────────────────────────────────────────
    if use_robot:
        reachy = ReachySDK(host='10.42.0.1')
        move_to_xyz(reachy, init_xyz, ocfg['gripper_deg'])
    else:
        reachy = None

    # ── 5. Buffers & timing ────────────────────────────────────────────
    pred_buf = np.zeros((seq_len, 3))
    obs_buf  = np.zeros((seq_len, 3))
    dt       = 1.0 / ocfg['fps']
    print(f"[RUN] {mtype.upper()} | T={seq_len} | fb={fb_flag} | robot={use_robot}")
    
    error_buf = np.zeros((seq_len, 3))

    # seed for sim‑mode
    pred_prev_xyz = init_xyz.copy()
    
    obs_xyz = np.zeros(3)
    error_hs    = None
    future_buf = np.full((seq_len, 100, 3), np.nan, dtype=np.float32)
    time.sleep(3)

    # ── 6. Main loop ────────────────────────────────────────────────────
    # for t in range(seq_len):
    for t in range(seq_len):
        tic = time.time()

        # 6‑1) observation ------------------------------------------------
        if use_robot:
            fk = reachy.l_arm.forward_kinematics()
            obs_xyz = fk[:3, 3].copy()
        else:
            obs_xyz = pred_prev_xyz
            
        # 6-2) normalise observation for network model --------------------
        obs_buf[t] = obs_xyz
        obs_norm = normalise(obs_xyz, dmin, dmax)
        obs_ts   = torch.tensor([obs_norm], dtype=torch.float32)

        # 6‑3) model step --------------------------------------------------
        print("--------------")
        print(
            "step: ", t,
            "\npred_prev_xyz: ", pred_prev_xyz,
            "\nobs_xyz: ", obs_xyz,
            "\nerr: ", pred_prev_xyz - obs_xyz
            )
        
        pred_prev_norm = normalise(pred_prev_xyz, dmin, dmax)
        pred_prev_ts   = torch.tensor([pred_prev_norm], dtype=torch.float32)
        
        # one-step-prediction
        x_pred, future_traj, error_h = net.step(t, obs_ts, pred_prev_ts, feedback=fb_flag)
        pred_xyz  = denormalise(x_pred.squeeze(0).numpy(), dmin, dmax)
        
        if error_hs is None:
            error_hs = np.empty((seq_len, len(error_h)), dtype=np.float32)
        error_hs[t] = np.array([torch.norm(e).item() if e is not None else 0.0 for e in error_h])
        
        future_xyz = denormalise(future_traj.squeeze(1).cpu().numpy(), dmin, dmax)  # shape (horizon, 3)
        horizon_t  = future_xyz.shape[0]
        if horizon_t > 100:
            raise RuntimeError(f"future_traj length {horizon_t} exceeds MAX_FUTURE_HORIZON 100")
        future_buf[t, :horizon_t] = future_xyz 
        
        print("predicted next step: ", pred_xyz)
        
        pred_buf[t] = pred_xyz
        pred_prev_xyz = pred_xyz
        error_buf[t] = pred_xyz - obs_xyz 
        
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
        if use_robot:
            tf = np.array([[-0.029, 0.284,-0.959, pred_xyz[0]],
                           [ 0.024, 0.959, 0.283, pred_xyz[1]],
                           [ 0.999,-0.015,-0.035, pred_xyz[2]],
                           [ 0.   , 0.   , 0.   , 1.]])
            try:
                q = reachy.l_arm.inverse_kinematics(tf)
                goto({j: p for j, p in zip(reachy.l_arm.joints.values(), q)},
                     duration=dt, interpolation_mode=InterpolationMode.MINIMUM_JERK)
            except Exception as e:
                print(f"[WARN] IK fail @t={t}: {e}")
            if hasattr(reachy.l_arm, 'l_gripper'):
                reachy.l_arm.l_gripper.goal_position = ocfg['gripper_deg']

        # 6‑4) keep fps ----------------------------------------------------
        time.sleep(max(0, dt - (time.time() - tic)))

    print("[DONE] sequence finished")
    
    # ── エラー統計を計算 ─────────────────────────────────────────
    error_mean = error_buf.mean(axis=0)
    error_cov  = np.cov(error_buf.T)
    error_l2   = np.linalg.norm(error_buf, axis=1)

    print("\n[ERROR STATS]")
    print(f"  mean error vector     : {error_mean} (m)")
    print(f"  covariance matrix (m²):\n{error_cov}")
    print(f"  L2 norm mean          : {error_l2.mean():.6f} m")
    print(f"  L2 norm std.dev       : {error_l2.std():.6f} m")
    print(f"  L2 norm 95‑percentile : {np.quantile(error_l2, 0.95):.6f} m")

    # ── 7. Save ----------------------------------------------------------
    save_root = resolve_path(ocfg['SequenceGeneration_save_directory'])
    os.makedirs(save_root, exist_ok=True)
    tag = f"class{ocfg['class_id']:02d}_" + \
        __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
    np.save(os.path.join(save_root, f'pred_xyz_{tag}.npy'), pred_buf)
    np.save(os.path.join(save_root, f'obs_xyz_{tag}.npy'),  obs_buf)
    print(f"[INFO] trajectories saved → {save_root}")
    np.save(os.path.join(save_root, f'error_hs_{tag}.npy'),  error_hs)
    np.save(os.path.join(save_root, f'future_xyz_{tag}.npy'), future_buf)

    if use_robot:
        time.sleep(5)
        reachy.turn_off_smoothly('l_arm')


# ╭────────────────────────── CLI ───────────────────────────────────────╮
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pcrnn_config', required=True, help='Path to TOML configuration')
    try:
        p.add_argument('--robot', action=argparse.BooleanOptionalAction, default=False,
                       help='Control physical Reachy (default True). "--no-robot" for pure simulation.')
    except AttributeError:
        def _str2bool(v):
            if isinstance(v, bool): return v
            if v.lower() in {'yes','true','t','y','1'}: return True
            if v.lower() in {'no','false','f','n','0'}: return False
            raise argparse.ArgumentTypeError('Boolean value expected.')
        p.add_argument('--robot', type=_str2bool, default=True)
    args = p.parse_args()
    run_online(args.pcrnn_config, use_robot=args.robot)