import numpy as np
import time
import argparse
import os

from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode

def move_to_home(reachy):
    home_position = np.array([
      [-0.029,  0.284, -0.959,  0.346],
      [ 0.024,  0.959,  0.283,  0.247],  
      [ 0.999, -0.015, -0.035, -0.319],
      [0, 0, 0, 1],  
    ])
    
    joint_pos_home = reachy.l_arm.inverse_kinematics(home_position)
    
    print("Moving to home position...")
    reachy.turn_on('l_arm')
    goto({joint: pos for joint, pos in zip(reachy.l_arm.joints.values(), joint_pos_home)}, duration=1.0, interpolation_mode=InterpolationMode.MINIMUM_JERK)

    if hasattr(reachy.l_arm, 'l_gripper'):
        reachy.l_arm.l_gripper.goal_position = 10.0

def playback_data(data_directory):
    filepath = data_directory
    data = np.load(filepath)  # shape = (num_seq, seq_len, 3)
    
    num_seq, seq_len, _ = data.shape
    fps = 20
    interval = 1.0 / fps
    
    # Connect to Reachy
    reachy = ReachySDK(host='10.42.0.1')
    reachy.turn_on('l_arm')

    print(f"Loaded data shape: {data.shape}")
    
    for seq in range(num_seq):
        print(f"\n[Playback] Sequence {seq+1}/{num_seq}")
        move_to_home(reachy)
        time.sleep(2)
        input("Press Enter to start playback...")

        for t in range(seq_len):
            print(f"Frame {t+1}/{seq_len}")
            xyz = data[seq, t]
            pose = np.array([
                [-0.029,  0.284, -0.959, xyz[0]],
                [ 0.024,  0.959,  0.283, xyz[1]],
                [ 0.999, -0.015, -0.035, xyz[2]],
                [0, 0, 0, 1],
            ])

            try:
                joint_angles = reachy.l_arm.inverse_kinematics(pose)
                goto({joint: pos for joint, pos in zip(reachy.l_arm.joints.values(), joint_angles)}, duration=interval, interpolation_mode=InterpolationMode.MINIMUM_JERK)
            except Exception as e:
                print(f"[WARN] IK failed at frame {t}: {e}")
            
            if hasattr(reachy.l_arm, 'l_gripper'):
                reachy.l_arm.l_gripper.goal_position = 10.0

    print("Playback complete.")
    reachy.turn_off_smoothly('l_arm')

def main():
    parser = argparse.ArgumentParser(description="Playback Reachy Trajectory")
    parser.add_argument("--data_directory", type=str, required=True, help="Directory containing the .npy trajectory file")
    args = parser.parse_args()
    
    playback_data(args.data_directory)

if __name__ == "__main__":
    main()
